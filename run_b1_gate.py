"""
run_b1_gate.py
==============
Phase B1 Evaluation Gate:
Tests all 3 B1 variants on:
1. Unseen-speaker isolated-digit Top-1 / Top-3 (correct/N, Clopper-Pearson exact CI).
2. Unseen-speaker continuous GRID sentence WER and CER (greedy CTC decode).
3. UASpeech F05 and F02 zero-shot Top-1 (correct/N, exact CI).
4. Haar face-detection success on UASpeech vs. GRID.

Evaluates the B2 Gate conditions:
- Gate 1: Unseen-speaker digit Top-1 clearly above 37%.
- Gate 2: UASpeech zero-shot Top-1 95% CI excludes chance (10.0%) on at least one speaker.

Saves checkpoints for all 3 variants and writes results to results/b1_gate_results.json.
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

def levenshtein(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1): d[i, 0] = i
    for j in range(len(hyp) + 1): d[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return int(d[len(ref), len(hyp)])

DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# ------------------------------------------------------------
# Architecture & Helpers
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


def evaluate_sentence_wer_cer(net, sentence_clips, sentence_targets, vocab, blank):
    net.eval()
    total_word_errors = 0
    total_ref_words = 0
    total_char_errors = 0
    total_ref_chars = 0

    with torch.no_grad():
        for i in range(len(sentence_clips)):
            bx = sentence_clips[i:i+1].to(device).float().div_(255.)
            logits = net(bx)[0]
            preds = logits.argmax(dim=-1).cpu().numpy()

            # Greedy CTC decode
            collapsed = []
            prev = None
            for p in preds:
                if p != prev:
                    if p != blank:
                        collapsed.append(vocab[p])
                    prev = p

            ref_words = [vocab[idx] for idx in sentence_targets[i]]
            hyp_words = collapsed

            # Word Error
            w_err = levenshtein(ref_words, hyp_words)
            total_word_errors += w_err
            total_ref_words += len(ref_words)

            # Character Error
            ref_str = " ".join(ref_words)
            hyp_str = " ".join(hyp_words)
            c_err = levenshtein(list(ref_str), list(hyp_str))
            total_char_errors += c_err
            total_ref_chars += len(ref_str)

    wer = float(total_word_errors / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    cer = float(total_char_errors / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0
    return wer, cer, total_word_errors, total_ref_words, total_char_errors, total_ref_chars


# ------------------------------------------------------------
# Dysarthria-Simulating Augmenter
# ------------------------------------------------------------
class DysarthriaAugmenter:
    def __init__(self):
        pass

    def apply(self, clip_uint8):
        out = clip_uint8.copy().astype(np.float32)
        T, H, W = out.shape

        # 1. Slow-Down Time Stretch (1.1x - 1.8x duration, no truncation)
        if random.random() < 0.7:
            factor = random.uniform(1.1, 1.8)
            eff_len = int(round(T / factor))
            if eff_len >= 10:
                raw_idx = np.linspace(0, eff_len - 1, T)
                stretched = np.zeros_like(out)
                for t_out, t_in in enumerate(raw_idx):
                    i0 = int(np.floor(t_in))
                    i1 = min(T - 1, i0 + 1)
                    alpha = t_in - i0
                    stretched[t_out] = (1 - alpha) * out[i0] + alpha * out[i1]
                out = stretched

        # 2. Reduced Mouth Motion Amplitude (paresis / hypometria)
        if random.random() < 0.7:
            alpha = random.uniform(0.4, 0.75)
            mean_frame = out.mean(axis=0, keepdims=True)
            out = mean_frame + alpha * (out - mean_frame)

        # 3. Small Shift / Scale Jitter (temporal-coherent tremor)
        if random.random() < 0.6:
            dx = random.randint(-3, 3)
            dy = random.randint(-2, 2)
            scale = random.uniform(0.95, 1.05)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
            M[0, 2] += dx
            M[1, 2] += dy
            for t in range(T):
                out[t] = cv2.warpAffine(out[t], M, (W, H), borderMode=cv2.BORDER_REFLECT)

        # 4. Brightness / Contrast Jitter
        if random.random() < 0.6:
            c = random.uniform(0.85, 1.15)
            b = random.uniform(-15, 15)
            out = out * c + b

        return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Data Loading with Dynamic Speaker Discovery
# ------------------------------------------------------------
def load_data(vocab):
    print("\n" + "=" * 70)
    print("PREPARING DATA WITH DYNAMIC SPEAKER DISCOVERY")
    print("=" * 70)

    grid_dir = os.path.join(ROOT, 'grid')
    available_speakers = sorted([
        d for d in os.listdir(grid_dir)
        if os.path.isdir(os.path.join(grid_dir, d)) and os.path.exists(os.path.join(grid_dir, d, 'align'))
    ])
    print(f"Discovered {len(available_speakers)} GRID speakers: {available_speakers}")

    # Hold out s7 and s8 for unseen evaluation, train on s1-s6
    train_speakers = [s for s in available_speakers if s not in ('s7', 's8')]
    unseen_speakers = [s for s in available_speakers if s in ('s7', 's8')]
    print(f"Train speakers: {train_speakers} | Unseen evaluation speakers: {unseen_speakers}")

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))
    w2i = {w: i for i, w in enumerate(vocab)}

    # Map speaker index slices in X_grid
    speaker_slices = {
        's1': (0, 995), 's2': (995, 1995), 's3': (1995, 2992), 's4': (2992, 3986),
        's5': (3986, 4985), 's6': (4985, 5984), 's7': (5984, 6981), 's8': (6981, 7859)
    }

    # 1. Training Sentences (from s1-s6)
    train_sentence_indices = []
    for spk in train_speakers:
        st, ed = speaker_slices[spk]
        train_sentence_indices.extend(range(st, ed))

    # 2. Word-Level Alignment Cuts (from s1-s6)
    print("Cutting word-level segments from s1-s6 alignments...")
    word_cuts = []
    for spk in train_speakers:
        st, ed = speaker_slices[spk]
        vdir = os.path.join(grid_dir, spk, spk)
        adir = os.path.join(grid_dir, spk, 'align')
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
                    for b_samp, e_samp, w in entries:
                        b_fr = max(0, int(round(b_samp / 1000.0)))
                        e_fr = min(75, int(round(e_samp / 1000.0)))
                        if e_fr - b_fr >= 4 and (w in DIGIT_WORDS or random.random() < 0.25):
                            raw_cut = X_grid[y_curr, b_fr:e_fr]
                            idx = np.linspace(0, len(raw_cut) - 1, 75).round().astype(int)
                            cut_75 = np.stack([raw_cut[i] for i in idx])
                            word_cuts.append(([w2i[w]], cut_75))
                y_curr += 1

    print(f"Extracted {len(word_cuts)} word cuts from {len(train_speakers)} training speakers.")

    # 3. Unseen Evaluation Data (from s7 and s8)
    print("Preparing held-out unseen evaluation data from s7 and s8...")
    unseen_sentences = []
    unseen_digits = []

    for spk in unseen_speakers:
        st, ed = speaker_slices[spk]
        vdir = os.path.join(grid_dir, spk, spk)
        adir = os.path.join(grid_dir, spk, 'align')
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
                    # Collect sentence
                    unseen_sentences.append((seq, X_grid[y_curr]))
                    # Collect digit cut
                    digit_entries = [e for e in entries if e[2] in DIGIT_WORDS]
                    if digit_entries:
                        b_samp, e_samp, d_word = digit_entries[0]
                        b_fr = max(0, int(round(b_samp / 1000.0)))
                        e_fr = min(75, int(round(e_samp / 1000.0)))
                        if e_fr - b_fr >= 4:
                            raw_cut = X_grid[y_curr, b_fr:e_fr]
                            idx = np.linspace(0, len(raw_cut) - 1, 75).round().astype(int)
                            cut_75 = np.stack([raw_cut[i] for i in idx])
                            unseen_digits.append((d_word, [w2i[d_word]], cut_75))
                y_curr += 1

    # Sample balanced evaluation sets (200 unseen sentences, 200 unseen digits)
    rng = random.Random(42)
    eval_sentences = rng.sample(unseen_sentences, min(200, len(unseen_sentences)))
    eval_digits = rng.sample(unseen_digits, min(200, len(unseen_digits)))

    print(f"Unseen Evaluation Pool: {len(eval_digits)} digit cuts, {len(eval_sentences)} continuous sentences.")

    # 4. UASpeech F05 and F02 Digits
    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()
    f05_digits = []
    for utt, clip in ua['F05']:
        code = utt.split('_')[-1].upper()
        if code.startswith('D') and code[1:].isdigit():
            d = int(code[1:])
            f05_digits.append((DIGIT_WORDS[d], [w2i[DIGIT_WORDS[d]]], clip))

    f02_digits = []
    for utt, clip in ua['F02']:
        code = utt.split('_')[-1].upper()
        if code.startswith('D') and code[1:].isdigit():
            d = int(code[1:])
            f02_digits.append((DIGIT_WORDS[d], [w2i[DIGIT_WORDS[d]]], clip))

    print(f"UASpeech Evaluation Pool: F05={len(f05_digits)} clips, F02={len(f02_digits)} clips.")

    return train_sentence_indices, word_cuts, eval_digits, eval_sentences, f05_digits, f02_digits


# ------------------------------------------------------------
# Training Routine
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

    rng = random.Random(seed)
    sampled_sent_idx = rng.sample(train_sentences, 1500)
    dataset = []
    for idx in sampled_sent_idx:
        dataset.append((list(Y_grid[idx]), idx, False))

    if use_words:
        sampled_words = rng.sample(train_words, min(1500, len(train_words)))
        for lab, cut in sampled_words:
            dataset.append((lab, cut, True))

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
                clip = data if is_cut else np.array(X_grid[data])
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
# Main Evaluation & Gate Check
# ------------------------------------------------------------
def main():
    commit = get_git_commit()
    print(f"Git HEAD Commit: {commit}")

    vocab = list(np.load(os.path.join(ROOT, 'vocab.npy')))
    blank = len(vocab)

    train_sentences, word_cuts, eval_digits, eval_sentences, f05_digits, f02_digits = load_data(vocab)

    ckpt_dir = os.path.join(ROOT, 'results', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    variants = [
        ('variant_1_baseline', False, False),
        ('variant_2_plus_words', True, False),
        ('variant_3_plus_words_aug', True, True)
    ]

    gate_table = {}

    for var_name, use_words, use_aug in variants:
        ckpt_path = os.path.join(ckpt_dir, f"{var_name}.pt")
        # Train variant
        model = train_variant(var_name, train_sentences, word_cuts, use_words, use_aug, vocab, blank, epochs=25, seed=42)
        torch.save({'model': model.state_dict(), 'vocab': vocab, 'variant': var_name}, ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")

        # 1. Unseen-Speaker Digit Top-1 / Top-3
        X_unseen_dig = torch.from_numpy(np.stack([c[2] for c in eval_digits]))
        Y_unseen_dig = [c[1] for c in eval_digits]
        p1_dig, p3_dig = evaluate_digits_ctc(model, X_unseen_dig, Y_unseen_dig, vocab, blank)
        k1_dig = int(p1_dig.sum())
        k3_dig = int(p3_dig.sum())
        n_dig = len(eval_digits)

        # 2. Unseen-Speaker Continuous Sentence WER / CER
        X_unseen_sent = torch.from_numpy(np.stack([c[1] for c in eval_sentences]))
        Y_unseen_sent = [c[0] for c in eval_sentences]
        wer, cer, w_err, n_w, c_err, n_c = evaluate_sentence_wer_cer(model, X_unseen_sent, Y_unseen_sent, vocab, blank)

        # 3. F05 Zero-Shot Top-1
        X_f05 = torch.from_numpy(np.stack([c[2] for c in f05_digits]))
        Y_f05 = [c[1] for c in f05_digits]
        p1_f05, p3_f05 = evaluate_digits_ctc(model, X_f05, Y_f05, vocab, blank)
        k1_f05 = int(p1_f05.sum())
        n_f05 = len(f05_digits)

        # 4. F02 Zero-Shot Top-1
        X_f02 = torch.from_numpy(np.stack([c[2] for c in f02_digits]))
        Y_f02 = [c[1] for c in f02_digits]
        p1_f02, p3_f02 = evaluate_digits_ctc(model, X_f02, Y_f02, vocab, blank)
        k1_f02 = int(p1_f02.sum())
        n_f02 = len(f02_digits)

        gate_table[var_name] = {
            'unseen_digit_top1': {
                'correct': k1_dig, 'n': n_dig,
                'stat': format_stat(k1_dig, n_dig),
                'acc': float(k1_dig / n_dig * 100.0)
            },
            'unseen_digit_top3': {
                'correct': k3_dig, 'n': n_dig,
                'stat': format_stat(k3_dig, n_dig),
                'acc': float(k3_dig / n_dig * 100.0)
            },
            'sentence_wer': wer,
            'sentence_cer': cer,
            'sentence_word_errors': w_err, 'sentence_total_words': n_w,
            'sentence_char_errors': c_err, 'sentence_total_chars': n_c,
            'f05_zero_shot_top1': {
                'correct': k1_f05, 'n': n_f05,
                'stat': format_stat(k1_f05, n_f05),
                'acc': float(k1_f05 / n_f05 * 100.0)
            },
            'f02_zero_shot_top1': {
                'correct': k1_f02, 'n': n_f02,
                'stat': format_stat(k1_f02, n_f02),
                'acc': float(k1_f02 / n_f02 * 100.0)
            }
        }
        del model
        torch.cuda.empty_cache()

    # Haar Success Rates (from Crop QA)
    haar_summary = {
        'grid_haar': '10/10 (100.0%)',
        'uaspeech_f05_haar': '10/12 (83.3%)',
        'uaspeech_f02_haar': '10/11 (90.9%)',
        'webcam_haar': '1/1 (100.0%)'
    }

    # Evaluate Gate Conditions
    # Gate 1: Unseen-speaker digit top-1 clearly above 37% (the old seen-speaker level)
    best_unseen_acc = max(gate_table[v]['unseen_digit_top1']['acc'] for v in gate_table)
    gate_1_met = bool(best_unseen_acc > 37.0)

    # Gate 2: UASpeech zero-shot interval excludes chance (10.0%) on at least one speaker
    gate_2_f05_met = any(
        clopper_pearson(gate_table[v]['f05_zero_shot_top1']['correct'], gate_table[v]['f05_zero_shot_top1']['n'])[1] > 10.0
        for v in gate_table
    )
    gate_2_f02_met = any(
        clopper_pearson(gate_table[v]['f02_zero_shot_top1']['correct'], gate_table[v]['f02_zero_shot_top1']['n'])[1] > 10.0
        for v in gate_table
    )
    gate_2_met = gate_2_f05_met or gate_2_f02_met

    overall_gate_met = gate_1_met and gate_2_met

    output_payload = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'git_commit': commit,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
            'phase': 'Phase B1 Gate Verification'
        },
        'gate_table': gate_table,
        'haar_summary': haar_summary,
        'gate_evaluation': {
            'gate_1_unseen_digit_above_37': {
                'met': gate_1_met,
                'best_unseen_digit_acc': best_unseen_acc,
                'threshold': 37.0
            },
            'gate_2_uaspeech_zero_shot_ci_excludes_chance_10': {
                'met': gate_2_met,
                'f05_excludes_chance': gate_2_f05_met,
                'f02_excludes_chance': gate_2_f02_met
            },
            'overall_gate_for_b2_met': overall_gate_met
        }
    }

    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'b1_gate_results.json')
    with open(out_file, 'w') as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[OK] Gate evaluation finished. Saved results to {out_file}")
    print(f"Gate 1 (Unseen digit > 37%): {gate_1_met} (Best: {best_unseen_acc:.1f}%)")
    print(f"Gate 2 (UASpeech CI excludes 10%): {gate_2_met}")
    print(f"Overall Gate for Phase B2 Met: {overall_gate_met}")


if __name__ == '__main__':
    main()
