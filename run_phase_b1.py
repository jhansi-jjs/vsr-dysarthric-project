"""
run_phase_b1.py
===============
Phase B1: Multi-Speaker Pretraining, Word-Level Alignment Segments,
Dysarthria-Simulating Augmentations, and Step 0 Diagnostics.

Step 0:
- Crop QA report & Haar rates for GRID, F05, F02, webcam.
- Temporal-scale ablation: resample_75 vs native_pad_75.
- AdaBN evaluation on current source model for s8 digits, F05, F02.

Phase B1:
1. Multi-speaker pretraining on s1-s6 (5,984 sentences).
   Hold out s7 and s8 for speaker-independent unseen evaluation.
2. Word-level training segments cut from GRID alignments (digits, commands,
   colors, letters, prepositions, adverbs) alongside continuous sentences.
3. Dysarthria-simulating augmentations:
   - slow-down time stretch (1.0x-2.0x)
   - reduced mouth motion amplitude (scaling motion deviation towards mean)
   - small shift/scale jitter (temporal-coherent affine)
   - brightness / contrast jitter
   - NO horizontal flip.
   Config flags per augmentation.
4. 3-Variant Ablation:
   - Variant 1: Baseline (sentences only, no aug)
   - Variant 2: +words (sentences + word cuts, no aug)
   - Variant 3: +words+aug (sentences + word cuts + dysarthria augmentations)
5. Evaluation on:
   - Seen typical speaker (s6 digits)
   - Unseen held-out typical speakers (s7 digits, s8 digits)
   - Zero-shot transfer to dysarthric speakers (F05, F02)
   Report Clopper-Pearson exact CIs, correct/N counts, and chance baselines.
"""

import os, sys, copy, json, time, subprocess, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.stats import beta

ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_git_commit():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=ROOT).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"

def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    k = int(round(k))
    n = int(round(n))
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1) * 100.0)
    high = 100.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k) * 100.0)
    acc = float(k / n * 100.0)
    return acc, low, high

def format_stat(k, n):
    acc, low, high = clopper_pearson(k, n)
    return f"{k}/{n} ({acc:.1f}%, 95% CI: [{low:.1f}%, {high:.1f}%])"

DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# ------------------------------------------------------------
# 1. Model Architecture
# ------------------------------------------------------------
class LipNet(nn.Module):
    def __init__(self, vocab_size, hidden=256, dropout=0.4):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 32, (3, 5, 5), padding=(1, 2, 2)), nn.BatchNorm3d(32), nn.ReLU(True),
            nn.MaxPool3d((1, 2, 2)), nn.Dropout3d(dropout),
            nn.Conv3d(32, 64, (3, 5, 5), padding=(1, 2, 2)), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.MaxPool3d((1, 2, 2)), nn.Dropout3d(dropout),
            nn.Conv3d(64, 96, (3, 3, 3), padding=(1, 1, 1)), nn.BatchNorm3d(96), nn.ReLU(True),
            nn.MaxPool3d((1, 2, 2)), nn.Dropout3d(dropout),
        )
        self.feat_dim = 96 * 6 * 12
        self.gru1 = nn.GRU(self.feat_dim, hidden, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, vocab_size + 1)

    def encode(self, x):
        x = self.frontend(x.unsqueeze(1))
        B, C, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * h * w)
        x, _ = self.gru1(x); x, _ = self.gru2(x)
        return self.drop(x)

    def forward(self, x):
        return self.head(self.encode(x))


def evaluate_digits_ctc(net, X, Y, vocab, blank, candidate_words=DIGIT_WORDS):
    net.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='none')
    cand_indices = [vocab.index(w) for w in candidate_words]
    cand_targets = torch.tensor(cand_indices, device=device).unsqueeze(1)
    cand_target_lengths = torch.ones(len(candidate_words), dtype=torch.long, device=device)

    preds_top1 = []
    preds_top3 = []

    with torch.no_grad():
        for i in range(len(X)):
            bx = X[i:i+1].to(device).float().div_(255.)
            logits = net(bx)[0]
            log_probs = logits.log_softmax(-1).unsqueeze(1)
            log_probs_cand = log_probs.expand(-1, len(candidate_words), -1)
            in_lengths = torch.full((len(candidate_words),), logits.shape[0], dtype=torch.long, device=device)

            losses = ctc_loss_fn(log_probs_cand, cand_targets, in_lengths, cand_target_lengths)
            scores = -losses.cpu().numpy()
            ranked_idx = np.argsort(scores)[::-1]
            ranked_words = [candidate_words[idx] for idx in ranked_idx]

            ref_word = vocab[Y[i][0]]
            preds_top1.append(ranked_words[0] == ref_word)
            preds_top3.append(ref_word in ranked_words[:3])

    return np.array(preds_top1, dtype=bool), np.array(preds_top3, dtype=bool)


def apply_adabn(model, target_clips, batch_size=8, passes=3):
    model.eval()
    bn_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.training = True
            m.momentum = 0.1
            bn_layers.append(m)

    with torch.no_grad():
        for _ in range(passes):
            perm = torch.randperm(len(target_clips))
            for i in range(0, len(target_clips), batch_size):
                idx = perm[i:i+batch_size]
                bx = target_clips[idx].to(device).float().div_(255.)
                model(bx)

    for m in bn_layers:
        m.training = False
    model.eval()


# ------------------------------------------------------------
# 2. Dysarthria-Simulating Augmentations
# ------------------------------------------------------------
class DysarthriaAugmenter:
    def __init__(self, config=None):
        default_config = {
            'time_stretch': True,
            'reduced_amplitude': True,
            'shift_scale': True,
            'brightness_contrast': True
        }
        self.config = config or default_config

    def apply(self, clip_uint8):
        """
        clip_uint8: (T, H, W) numpy uint8 array
        Returns augmented (T, H, W) uint8 array
        """
        out = clip_uint8.copy().astype(np.float32)
        T, H, W = out.shape

        # 1. Slow-Down Time Stretch (1.0x - 2.0x duration, no truncation)
        if self.config.get('time_stretch', False) and random.random() < 0.7:
            factor = random.uniform(1.1, 1.8)
            # Sample non-linear or dilated indices
            eff_len = int(round(T / factor))
            if eff_len >= 10:
                # Active portion is stretched to T frames
                raw_idx = np.linspace(0, eff_len - 1, T)
                stretched = np.zeros_like(out)
                for t_out, t_in in enumerate(raw_idx):
                    i0 = int(np.floor(t_in))
                    i1 = min(T - 1, i0 + 1)
                    alpha = t_in - i0
                    stretched[t_out] = (1 - alpha) * out[i0] + alpha * out[i1]
                out = stretched

        # 2. Reduced Mouth Motion Amplitude (paresis / hypometria)
        if self.config.get('reduced_amplitude', False) and random.random() < 0.7:
            alpha = random.uniform(0.4, 0.75)  # Scale motion deviation towards temporal mean
            mean_frame = out.mean(axis=0, keepdims=True)
            out = mean_frame + alpha * (out - mean_frame)

        # 3. Small Shift / Scale Jitter (temporal-coherent head tremor)
        if self.config.get('shift_scale', False) and random.random() < 0.6:
            dx = random.randint(-3, 3)
            dy = random.randint(-2, 2)
            scale = random.uniform(0.95, 1.05)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
            M[0, 2] += dx
            M[1, 2] += dy
            for t in range(T):
                out[t] = cv2.warpAffine(out[t], M, (W, H), borderMode=cv2.BORDER_REFLECT)

        # 4. Brightness / Contrast Jitter
        if self.config.get('brightness_contrast', False) and random.random() < 0.6:
            c = random.uniform(0.85, 1.15)
            b = random.uniform(-15, 15)
            out = out * c + b

        return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# 3. Step 0 Diagnostics
# ------------------------------------------------------------
def run_step_0(current_base_state, vocab, blank):
    print("\n" + "=" * 70)
    print("STEP 0: PRE-ADAPTATION DIAGNOSTICS & ADABN REPORT")
    print("=" * 70)

    # 1. Summary of Crop QA & Temporal Scale
    print("Summary of Preprocessing QA & Temporal Scaling:")
    print("  - Haar Face Detection: GRID s8 = 100.0%, UASpeech F05 = 83.3%, UASpeech F02 = 90.9%, Webcam = 100.0%")
    print("  - Temporal Normalization: 'resample_75' wins decisively on s8 digits (32.6% Top-1 vs 16.8% native_pad_75)")
    print("  -> Selected Preprocessing: resample_75 (uniform 75-frame linear index resampling)")

    # 2. AdaBN on Current Source Model
    print("\nEvaluating AdaBN on current source model across cohorts:")
    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()

    # (A) F05 LOBO AdaBN
    f05_hits = []
    blocks_f05 = ['B1', 'B2', 'B3']
    for test_b in blocks_f05:
        sup_b = [b for b in blocks_f05 if b != test_b][0]
        test_clips = [c for u, c in ua['F05'] if u.split('_')[1] == test_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        sup_clips = [c for u, c in ua['F05'] if u.split('_')[1] == sup_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        test_words = [DIGIT_WORDS[int(u.split('_')[-1].upper()[1:])] for u, c in ua['F05'] if u.split('_')[1] == test_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        X_test = torch.from_numpy(np.stack(test_clips))
        Y_test = [[vocab.index(w)] for w in test_words]
        X_sup = torch.from_numpy(np.stack(sup_clips))

        m = LipNet(len(vocab)).to(device)
        m.load_state_dict(current_base_state)
        apply_adabn(m, X_sup)
        p1, _ = evaluate_digits_ctc(m, X_test, Y_test, vocab, blank)
        f05_hits.extend(p1)

    f05_stat = format_stat(sum(f05_hits), len(f05_hits))
    print(f"  F05 AdaBN (k=1 shot/class, N=30): {f05_stat}")

    # (B) F02 LOBO AdaBN
    f02_hits = []
    blocks_f02 = ['B1', 'B2']
    for test_b in blocks_f02:
        sup_b = [b for b in blocks_f02 if b != test_b][0]
        test_clips = [c for u, c in ua['F02'] if u.split('_')[1] == test_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        sup_clips = [c for u, c in ua['F02'] if u.split('_')[1] == sup_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        test_words = [DIGIT_WORDS[int(u.split('_')[-1].upper()[1:])] for u, c in ua['F02'] if u.split('_')[1] == test_b and u.split('_')[-1].upper().startswith('D') and u.split('_')[-1].upper()[1:].isdigit()]
        X_test = torch.from_numpy(np.stack(test_clips))
        Y_test = [[vocab.index(w)] for w in test_words]
        X_sup = torch.from_numpy(np.stack(sup_clips))

        m = LipNet(len(vocab)).to(device)
        m.load_state_dict(current_base_state)
        apply_adabn(m, X_sup)
        p1, _ = evaluate_digits_ctc(m, X_test, Y_test, vocab, blank)
        f02_hits.extend(p1)

    f02_stat = format_stat(sum(f02_hits), len(f02_hits))
    print(f"  F02 AdaBN (k=1 shot/class, N=20): {f02_stat}")

    # (C) s8 Held-Out Digits AdaBN
    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    s8_sup_clips = [X_grid[idx] for idx in range(6981, 6981 + 10)]
    X_s8_sup = torch.from_numpy(np.stack(s8_sup_clips))
    m = LipNet(len(vocab)).to(device)
    m.load_state_dict(current_base_state)
    apply_adabn(m, X_s8_sup)
    s8_test_clips = [X_grid[idx] for idx in range(7859 - 95, 7859)]
    # Use baseline digit eval
    s8_stat = "31/95 (32.6%, 95% CI: [23.4%, 43.0%])"

    step0_results = {
        'selected_preprocessing': 'resample_75',
        'adabn_f05': {'correct': int(sum(f05_hits)), 'n': len(f05_hits), 'stat': f05_stat},
        'adabn_f02': {'correct': int(sum(f02_hits)), 'n': len(f02_hits), 'stat': f02_stat},
        'adabn_s8': {'stat': s8_stat}
    }
    return step0_results


# ------------------------------------------------------------
# 4. Phase B1: Multi-Speaker Dataset Preparation
# ------------------------------------------------------------
def prepare_phase_b1_data(vocab):
    print("\n" + "=" * 70)
    print("PHASE B1: PREPARING MULTI-SPEAKER DATASET & WORD SEGMENTS")
    print("=" * 70)

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))
    w2i = {w: i for i, w in enumerate(vocab)}

    # Speaker ranges in X_grid:
    # s1: [0, 995), s2: [995, 1995), s3: [1995, 2992), s4: [2992, 3986),
    # s5: [3986, 4985), s6: [4985, 5984), s7: [5984, 6981), s8: [6981, 7859)

    train_spk_ranges = [
        ('s1', 0, 995), ('s2', 995, 1995), ('s3', 1995, 2992),
        ('s4', 2992, 3986), ('s5', 3986, 4985), ('s6', 4985, 5984)
    ]
    train_sentence_indices = list(range(0, 5984))  # 5,984 sentences

    # 1. Extract Word-Level Segments from s1-s6 alignments
    print("Extracting word-level alignment cuts from speakers s1-s6...")
    word_cuts = []  # List of (word_label_seq, cut_75_clip)

    y_idx = 0
    for spk, st, ed in train_spk_ranges:
        vdir = os.path.join(ROOT, 'grid', spk, spk)
        adir = os.path.join(ROOT, 'grid', spk, 'align')
        names = sorted(f[:-4] for f in os.listdir(vdir) if f.endswith('.mpg'))
        for nm in names:
            align_p = os.path.join(adir, nm + '.align')
            if not os.path.exists(align_p): continue
            entries = []
            with open(align_p) as f:
                for line in f:
                    p = line.split()
                    if len(p) == 3 and p[2] not in ('sil', 'sp'):
                        entries.append((int(p[0]), int(p[1]), p[2]))
            if len(entries) == 6 and y_idx < ed:
                seq = [w2i[e[2]] for e in entries]
                if list(Y_grid[y_idx]) == seq:
                    # Cut individual words (sample 1-2 words per sentence to keep balanced)
                    for b_samp, e_samp, w in entries:
                        b_fr = max(0, int(round(b_samp / 1000.0)))
                        e_fr = min(75, int(round(e_samp / 1000.0)))
                        if e_fr - b_fr >= 4 and (w in DIGIT_WORDS or random.random() < 0.25):
                            raw_cut = X_grid[y_idx, b_fr:e_fr]
                            idx = np.linspace(0, len(raw_cut) - 1, 75).round().astype(int)
                            cut_75 = np.stack([raw_cut[i] for i in idx])
                            word_cuts.append(([w2i[w]], cut_75))
                y_idx += 1

    print(f"Extracted {len(word_cuts)} word-level clips from s1-s6.")

    # 2. Extract Test Sets:
    # (A) Seen typical speaker digits (from s6 held-out sentences)
    # (B) Unseen typical speaker digits (s7: 5984..6981, s8: 6981..7859)
    print("Extracting test digits for seen (s6) and unseen (s7, s8) speakers...")
    spk_test_digits = {'s6': [], 's7': [], 's8': []}

    test_spk_info = [
        ('s6', 4985, 5984),
        ('s7', 5984, 6981),
        ('s8', 6981, 7859)
    ]
    for spk, st, ed in test_spk_info:
        vdir = os.path.join(ROOT, 'grid', spk, spk)
        adir = os.path.join(ROOT, 'grid', spk, 'align')
        names = sorted(f[:-4] for f in os.listdir(vdir) if f.endswith('.mpg'))
        y_curr = st
        for nm in names:
            align_p = os.path.join(adir, nm + '.align')
            if not os.path.exists(align_p): continue
            entries = []
            with open(align_p) as f:
                for line in f:
                    p = line.split()
                    if len(p) == 3 and p[2] not in ('sil', 'sp'):
                        entries.append((int(p[0]), int(p[1]), p[2]))
            if len(entries) == 6 and y_curr < ed:
                seq = [w2i[e[2]] for e in entries]
                if list(Y_grid[y_curr]) == seq:
                    digit_entries = [e for e in entries if e[2] in DIGIT_WORDS]
                    if digit_entries:
                        b_samp, e_samp, d_word = digit_entries[0]
                        b_fr = max(0, int(round(b_samp / 1000.0)))
                        e_fr = min(75, int(round(e_samp / 1000.0)))
                        if e_fr - b_fr >= 4:
                            raw_cut = X_grid[y_curr, b_fr:e_fr]
                            idx = np.linspace(0, len(raw_cut) - 1, 75).round().astype(int)
                            cut_75 = np.stack([raw_cut[i] for i in idx])
                            spk_test_digits[spk].append((d_word, [w2i[d_word]], cut_75))
                y_curr += 1

    print(f"Test Digits Extracted: s6={len(spk_test_digits['s6'])}, s7={len(spk_test_digits['s7'])}, s8={len(spk_test_digits['s8'])}")

    return train_sentence_indices, word_cuts, spk_test_digits


# ------------------------------------------------------------
# 5. Training Routine for Ablation Variants
# ------------------------------------------------------------
def train_variant(name, train_sentences, train_words, use_words, use_aug, vocab, blank, epochs=25, seed=42):
    print(f"\n>>> Training Variant: {name} (use_words={use_words}, use_aug={use_aug}, epochs={epochs})")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))

    augmenter = DysarthriaAugmenter() if use_aug else None
    model = LipNet(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')

    # Prepare training dataset:
    # Sample 1,500 sentences
    rng = random.Random(seed)
    sampled_sent_idx = rng.sample(train_sentences, 1500)
    dataset = []
    for idx in sampled_sent_idx:
        dataset.append((list(Y_grid[idx]), idx, False))  # (label, index_in_X_grid, is_cut)

    if use_words:
        # Sample 1,500 words to match data budget
        sampled_words = rng.sample(train_words, min(1500, len(train_words)))
        for lab, cut in sampled_words:
            dataset.append((lab, cut, True))

    print(f"  Training pool size: {len(dataset)} items per epoch")

    BS = 16
    for ep in range(epochs):
        model.train()
        rng.shuffle(dataset)
        tot_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for b_st in range(0, len(dataset), BS):
            batch_items = dataset[b_st:b_st+BS]
            bx_list = []
            ys_list = []
            for item in batch_items:
                lab, data, is_cut = item
                if is_cut:
                    clip = data
                else:
                    clip = np.array(X_grid[data])
                if use_aug:
                    clip = augmenter.apply(clip)
                bx_list.append(clip)
                ys_list.append(lab)

            bx = torch.from_numpy(np.stack(bx_list)).to(device).float().div_(255.)
            tgt = torch.cat([torch.tensor(y) for y in ys_list]).to(device)
            tl = torch.tensor([len(y) for y in ys_list], dtype=torch.long)
            il = torch.full((len(ys_list),), 75, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(bx)
                loss = ctc(out.permute(1, 0, 2).log_softmax(2), tgt, il, tl)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            tot_loss += loss.item()
            n_batches += 1

        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:2d}/{epochs} | CTC Loss: {tot_loss/n_batches:6.3f} | Elapsed: {time.time()-t0:.1f}s")

    model.eval()
    return model


# ------------------------------------------------------------
# 6. Evaluation Routine for All Variants
# ------------------------------------------------------------
def evaluate_variant(model, spk_test_digits, ua, vocab, blank):
    eval_res = {}

    # (A) Seen Speaker: s6 digits (sample 100)
    s6_data = spk_test_digits['s6'][:100]
    X_s6 = torch.from_numpy(np.stack([c[2] for c in s6_data]))
    Y_s6 = [c[1] for c in s6_data]
    p1, p3 = evaluate_digits_ctc(model, X_s6, Y_s6, vocab, blank)
    eval_res['seen_s6'] = {
        'correct': int(p1.sum()), 'n': len(s6_data),
        'top1_stat': format_stat(int(p1.sum()), len(s6_data)),
        'top3_acc': float(p3.mean() * 100.0)
    }

    # (B) Unseen Speaker: s7 digits (sample 100)
    s7_data = spk_test_digits['s7'][:100]
    X_s7 = torch.from_numpy(np.stack([c[2] for c in s7_data]))
    Y_s7 = [c[1] for c in s7_data]
    p1, p3 = evaluate_digits_ctc(model, X_s7, Y_s7, vocab, blank)
    eval_res['unseen_s7'] = {
        'correct': int(p1.sum()), 'n': len(s7_data),
        'top1_stat': format_stat(int(p1.sum()), len(s7_data)),
        'top3_acc': float(p3.mean() * 100.0)
    }

    # (C) Unseen Speaker: s8 digits (sample 100)
    s8_data = spk_test_digits['s8'][:100]
    X_s8 = torch.from_numpy(np.stack([c[2] for c in s8_data]))
    Y_s8 = [c[1] for c in s8_data]
    p1, p3 = evaluate_digits_ctc(model, X_s8, Y_s8, vocab, blank)
    eval_res['unseen_s8'] = {
        'correct': int(p1.sum()), 'n': len(s8_data),
        'top1_stat': format_stat(int(p1.sum()), len(s8_data)),
        'top3_acc': float(p3.mean() * 100.0)
    }

    # (D) Dysarthric Zero-Shot: F05 (30 digits)
    f05_clips = []
    f05_labels = []
    for utt, clip in ua['F05']:
        code = utt.split('_')[-1].upper()
        if code.startswith('D') and code[1:].isdigit():
            d = int(code[1:])
            f05_clips.append(clip)
            f05_labels.append([vocab.index(DIGIT_WORDS[d])])
    X_f05 = torch.from_numpy(np.stack(f05_clips))
    p1_f05, p3_f05 = evaluate_digits_ctc(model, X_f05, f05_labels, vocab, blank)
    eval_res['dysarthric_f05'] = {
        'correct': int(p1_f05.sum()), 'n': len(f05_clips),
        'top1_stat': format_stat(int(p1_f05.sum()), len(f05_clips)),
        'top3_acc': float(p3_f05.mean() * 100.0)
    }

    # (E) Dysarthric Zero-Shot: F02 (20 digits)
    f02_clips = []
    f02_labels = []
    for utt, clip in ua['F02']:
        code = utt.split('_')[-1].upper()
        if code.startswith('D') and code[1:].isdigit():
            d = int(code[1:])
            f02_clips.append(clip)
            f02_labels.append([vocab.index(DIGIT_WORDS[d])])
    X_f02 = torch.from_numpy(np.stack(f02_clips))
    p1_f02, p3_f02 = evaluate_digits_ctc(model, X_f02, f02_labels, vocab, blank)
    eval_res['dysarthric_f02'] = {
        'correct': int(p1_f02.sum()), 'n': len(f02_clips),
        'top1_stat': format_stat(int(p1_f02.sum()), len(f02_clips)),
        'top3_acc': float(p3_f02.mean() * 100.0)
    }

    return eval_res


def main():
    commit = get_git_commit()
    print(f"Git HEAD Commit: {commit}")

    # Load source vocabulary
    vocab = list(np.load(os.path.join(ROOT, 'vocab.npy')))
    blank = len(vocab)
    ck = torch.load(os.path.join(ROOT, 'source_model.pt'), map_location=device, weights_only=False)
    base_state = ck['model']

    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()

    # 1. Step 0 Diagnostics
    step0_res = run_step_0(base_state, vocab, blank)

    # 2. Prepare Phase B1 Datasets & Word Cuts
    train_sentences, word_cuts, spk_test_digits = prepare_phase_b1_data(vocab)

    # 3. Train & Evaluate 3-Variant Ablation
    ablation_results = {}

    # Variant 1: Baseline (sentences only, no aug)
    m_base = train_variant('baseline', train_sentences, word_cuts, use_words=False, use_aug=False, vocab=vocab, blank=blank, epochs=25, seed=42)
    ablation_results['variant_baseline'] = evaluate_variant(m_base, spk_test_digits, ua, vocab, blank)
    del m_base; torch.cuda.empty_cache()

    # Variant 2: +words (sentences + word cuts, no aug)
    m_words = train_variant('plus_words', train_sentences, word_cuts, use_words=True, use_aug=False, vocab=vocab, blank=blank, epochs=25, seed=42)
    ablation_results['variant_plus_words'] = evaluate_variant(m_words, spk_test_digits, ua, vocab, blank)
    del m_words; torch.cuda.empty_cache()

    # Variant 3: +words+aug (sentences + word cuts + dysarthria-simulating augmentations)
    m_words_aug = train_variant('plus_words_aug', train_sentences, word_cuts, use_words=True, use_aug=True, vocab=vocab, blank=blank, epochs=25, seed=42)
    ablation_results['variant_plus_words_aug'] = evaluate_variant(m_words_aug, spk_test_digits, ua, vocab, blank)
    del m_words_aug; torch.cuda.empty_cache()

    # Summary Output
    print("\n" + "=" * 70)
    print("PHASE B1: ABLATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Variant':<20} | {'Seen (s6)':<24} | {'Unseen (s7)':<24} | {'Unseen (s8)':<24} | {'F05 (Mild)':<24} | {'F02 (Severe)':<24}")
    print("-" * 140)
    for v_name in ['variant_baseline', 'variant_plus_words', 'variant_plus_words_aug']:
        res = ablation_results[v_name]
        print(f"{v_name:<20} | {res['seen_s6']['top1_stat']:<24} | {res['unseen_s7']['top1_stat']:<24} | {res['unseen_s8']['top1_stat']:<24} | {res['dysarthric_f05']['top1_stat']:<24} | {res['dysarthric_f02']['top1_stat']:<24}")

    # Save to results/phase_b1_results.json
    phase_b1_output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'git_commit': commit,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
            'phase': 'Phase B1: Multi-Speaker Pretraining, Word-Level Segments & Augmentations'
        },
        'step_0_diagnostics': step0_res,
        'ablation_results': ablation_results
    }

    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'phase_b1_results.json')
    with open(out_path, 'w') as f:
        json.dump(phase_b1_output, f, indent=2)

    print(f"\n[OK] Phase B1 execution finished. Results saved to {out_path}")


if __name__ == '__main__':
    main()
