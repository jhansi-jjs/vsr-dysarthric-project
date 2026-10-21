"""
run_phase_a2.py
===============
Phase A2: Diagnosis and Evaluation Redesign.
- Control Test: Zero-shot evaluation on typical speaker s8 isolated digits.
- Per-class k-shot definition (k shots per class) with legacy 'total' flag.
- Leave-One-Block-Out (LOBO) cross-validation for F05 (3 folds) and F02 (2 folds).
- Non-parametric bootstrap 95% confidence intervals over test clips.
- Chance and majority-class baselines.
- Prototype baseline: Frozen LipNet encoder embeddings + cosine similarity.
- Extended 155 repeated words classification across blocks via prototype classifier.
- Drop redundant constrained WER from tables.
- Save everything under results/ with git commit, config, and metrics.
"""

import os, sys, copy, json, time, subprocess
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_git_commit():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=ROOT).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"

# ------------------------------------------------------------
# 1. Models & Adaptation
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


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.ReLU(True)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class PersonalizedLipNet(nn.Module):
    def __init__(self, base, mode='adapter'):
        super().__init__()
        self.base = base
        self.mode = mode
        dim = base.head.in_features
        self.adapter = Adapter(dim).to(next(base.parameters()).device) if mode == 'adapter' else None
        if mode == 'adapter':
            for p in self.base.parameters(): p.requires_grad = False
            for p in self.base.head.parameters(): p.requires_grad = True

    def forward(self, x):
        z = self.base.encode(x)
        if self.adapter is not None: z = self.adapter(z)
        return self.base.head(z)


def personalize(base_state, Xs, Ys, mode, vocab, blank, epochs=40, lr=1e-4):
    base = LipNet(len(vocab)).to(device)
    base.load_state_dict(base_state)
    net = PersonalizedLipNet(base, mode=mode).to(device) if mode != 'full' else base
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    net.train()
    for _ in range(epochs):
        for i in range(0, len(Xs), 2):
            bx = Xs[i:i+2].to(device).float().div_(255.)
            ys = Ys[i:i+2]
            tgt = torch.cat([torch.tensor(t) for t in ys]).to(device)
            tl = torch.tensor([len(t) for t in ys], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            out = net(bx)
            il = torch.full((len(ys),), out.shape[1], dtype=torch.long)
            loss = ctc(out.log_softmax(2).permute(1, 0, 2), tgt, il, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    net.eval()
    return net


# ------------------------------------------------------------
# 2. Decoding & Evaluation Helpers
# ------------------------------------------------------------
DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def evaluate_digits_ctc(net, X, Y, vocab, blank, candidate_words=DIGIT_WORDS):
    """Closed-set CTC single-word scoring for digit classification."""
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


def extract_embeddings(model, clips, batch_size=16):
    """Extract normalized mean-pooled temporal embeddings from frozen encoder."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i+batch_size].to(device).float().div_(255.)
            z = model.encode(batch)  # (B, T, 512)
            # Global average pool over time
            emb = z.mean(dim=1)  # (B, 512)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def evaluate_prototypes(support_embs, support_labels, test_embs, test_labels):
    """
    Nearest Class Prototype (Cosine Similarity).
    support_embs: (N_sup, D), support_labels: list of class IDs (str or int)
    test_embs: (N_test, D), test_labels: list of class IDs
    """
    classes = sorted(list(set(support_labels)))
    prototypes = []
    for c in classes:
        c_idx = [i for i, lab in enumerate(support_labels) if lab == c]
        proto = support_embs[c_idx].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
    prototypes = np.stack(prototypes)  # (C, D)

    # Cosine similarities: (N_test, C)
    sims = np.dot(test_embs, prototypes.T)

    preds_top1 = []
    preds_top3 = []
    for i, true_lab in enumerate(test_labels):
        ranked_classes = [classes[idx] for idx in np.argsort(sims[i])[::-1]]
        preds_top1.append(ranked_classes[0] == true_lab)
        preds_top3.append(true_lab in ranked_classes[:3])

    return np.array(preds_top1, dtype=bool), np.array(preds_top3, dtype=bool)


def bootstrap_ci(binary_correct_array, n_boot=2000, alpha=0.05, seed=42):
    """Compute non-parametric bootstrap 95% confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(binary_correct_array)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_accs = []
    for _ in range(n_boot):
        sample = rng.choice(binary_correct_array, size=n, replace=True)
        boot_accs.append(np.mean(sample) * 100.0)
    low = float(np.percentile(boot_accs, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_accs, 100 * (1.0 - alpha / 2.0)))
    mean_acc = float(np.mean(binary_correct_array) * 100.0)
    return mean_acc, low, high


# ------------------------------------------------------------
# 3. Execution Routines
# ------------------------------------------------------------
def run_control_test(model, vocab, blank):
    """
    Control Test: Cut isolated digit clips from typical speaker s8
    using alignment timings and evaluate zero-shot.
    """
    print("\n" + "=" * 70)
    print("1. CONTROL TEST: TYPICAL SPEAKER ISOLATED DIGIT ACCURACY")
    print("=" * 70)

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))
    w2i = {w: i for i, w in enumerate(vocab)}

    s8_adir = os.path.join(ROOT, 'grid', 's8', 'align')
    s8_vdir = os.path.join(ROOT, 'grid', 's8', 's8')
    s8_names = sorted(f[:-4] for f in os.listdir(s8_vdir) if f.endswith('.mpg'))

    def read_alignment(path):
        out = []
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 3: out.append((int(p[0]), int(p[1]), p[2]))
        return out

    def resample_to(frames, n=75):
        if len(frames) == 0: return None
        idx = np.linspace(0, len(frames)-1, n).round().astype(int)
        return np.stack([frames[i] for i in idx])

    s8_cuts = []
    y_idx = 6981  # Start index of s8 in X_grid
    for nm in s8_names:
        align_p = os.path.join(s8_adir, nm + '.align')
        entries = read_alignment(align_p)
        words = [e[2] for e in entries if e[2] not in ('sil', 'sp')]
        if len(words) == 6 and y_idx < len(Y_grid):
            seq = [w2i[w] for w in words]
            if list(Y_grid[y_idx]) == seq:
                digit_entries = [e for e in entries if e[2] in DIGIT_WORDS]
                if digit_entries:
                    b_samp, e_samp, d_word = digit_entries[0]
                    b_fr = max(0, int(round(b_samp / 1000.0)))
                    e_fr = min(75, int(round(e_samp / 1000.0)))
                    if e_fr - b_fr >= 4:
                        raw_cut = X_grid[y_idx, b_fr:e_fr]
                        cut_75 = resample_to(raw_cut, 75)
                        s8_cuts.append((d_word, cut_75))
                y_idx += 1

    X_cuts = torch.from_numpy(np.stack([c[1] for c in s8_cuts]))
    Y_cuts = [[vocab.index(c[0])] for c in s8_cuts]

    p1, p3 = evaluate_digits_ctc(model, X_cuts, Y_cuts, vocab, blank)
    mean_top1, ci_low, ci_high = bootstrap_ci(p1)
    mean_top3 = float(np.mean(p3) * 100.0)

    # Class balance / majority class in control set
    labels_list = [c[0] for c in s8_cuts]
    _, counts = np.unique(labels_list, return_counts=True)
    majority_class_acc = float(np.max(counts) / len(labels_list) * 100.0)
    chance_level = 10.0  # 1/10

    control_result = {
        'speaker': 's8 (Typical GRID speaker)',
        'n_clips': len(s8_cuts),
        'top1_acc': mean_top1,
        'top1_ci95': [ci_low, ci_high],
        'top3_acc': mean_top3,
        'chance_level': chance_level,
        'majority_class_acc': majority_class_acc,
        'uaspeech_control_video': 'None available (UASpeech archives contain only F05, F02)'
    }

    print(f"Typical Speaker s8 (Isolated Digits, N={len(s8_cuts)}):")
    print(f"  Top-1 Accuracy: {mean_top1:.1f}% (95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])")
    print(f"  Top-3 Accuracy: {mean_top3:.1f}%")
    print(f"  Chance Level:   {chance_level:.1f}%")
    print(f"  Majority Class: {majority_class_acc:.1f}%")
    print(f"  UASpeech Control Videos: None available locally (F05 & F02 only).")

    return control_result


def run_lobo_digit_experiments(ua, base_state, vocab, blank):
    """
    Leave-One-Block-Out (LOBO) evaluation on 10 digits for F05 (3 folds) and F02 (2 folds).
    Evaluates:
      - Zero-shot LipNet
      - Full Fine-Tuning (k=1 shot/class, 10 clips)
      - Adapter Tuning (k=1 shot/class, 10 clips)
      - Prototype Baseline (k=1 shot/class, 10 clips)
      - Prototype Baseline (k=2 shots/class, 20 clips, F05 only)
    """
    print("\n" + "=" * 70)
    print("2. LEAVE-ONE-BLOCK-OUT (LOBO) DIGIT EVALUATION (k=1 SHOT/CLASS)")
    print("=" * 70)

    results_digits = {}

    for spk, blocks in [('F05', ['B1', 'B2', 'B3']), ('F02', ['B1', 'B2'])]:
        desc = "Mild dysarthria (95% intelligibility)" if spk == 'F05' else "Severe dysarthria (29% intelligibility)"
        print(f"\n>>> SPEAKER: {spk} ({desc}) — LOBO Folds: {blocks}")

        # Extract digit clips per block
        spk_digit_blocks = {}
        for b in blocks:
            spk_digit_blocks[b] = []
            for utt, clip in ua[spk]:
                parts = utt.split('_')
                if parts[1] == b:
                    code = parts[-1].upper()
                    if code.startswith('D') and code[1:].isdigit():
                        digit_val = int(code[1:])
                        spk_digit_blocks[b].append({
                            'utt': utt, 'block': b, 'digit': digit_val,
                            'word': DIGIT_WORDS[digit_val],
                            'label': [vocab.index(DIGIT_WORDS[digit_val])],
                            'clip': clip
                        })

        methods = ['zero_shot', 'prototype_k1', 'adapter_k1', 'full_k1']
        if spk == 'F05':
            methods.append('prototype_k2')

        method_preds_top1 = {m: [] for m in methods}
        method_preds_top3 = {m: [] for m in methods}

        # Leave-One-Block-Out Loop
        for fold_idx, test_b in enumerate(blocks):
            support_blocks = [b for b in blocks if b != test_b]
            test_items = spk_digit_blocks[test_b]
            X_test = torch.from_numpy(np.stack([item['clip'] for item in test_items]))
            Y_test = [item['label'] for item in test_items]
            labels_test_words = [item['word'] for item in test_items]

            # 1. Zero-shot
            base_model = LipNet(len(vocab)).to(device)
            base_model.load_state_dict(base_state)
            p1, p3 = evaluate_digits_ctc(base_model, X_test, Y_test, vocab, blank)
            method_preds_top1['zero_shot'].extend(p1)
            method_preds_top3['zero_shot'].extend(p3)

            # 2. Support data for k=1 shot/class (10 clips total, 1 of each digit)
            sup_items_k1 = spk_digit_blocks[support_blocks[0]]  # Exactly 1 of each digit 0-9
            X_sup_k1 = torch.from_numpy(np.stack([item['clip'] for item in sup_items_k1]))
            Y_sup_k1 = [item['label'] for item in sup_items_k1]
            labels_sup_k1_words = [item['word'] for item in sup_items_k1]

            # 3. Prototype Classifier (k=1)
            test_embs = extract_embeddings(base_model, X_test)
            sup_embs_k1 = extract_embeddings(base_model, X_sup_k1)
            p1_proto, p3_proto = evaluate_prototypes(sup_embs_k1, labels_sup_k1_words, test_embs, labels_test_words)
            method_preds_top1['prototype_k1'].extend(p1_proto)
            method_preds_top3['prototype_k1'].extend(p3_proto)

            # 4. Prototype Classifier (k=2, F05 only)
            if spk == 'F05':
                sup_items_k2 = spk_digit_blocks[support_blocks[0]] + spk_digit_blocks[support_blocks[1]]
                X_sup_k2 = torch.from_numpy(np.stack([item['clip'] for item in sup_items_k2]))
                labels_sup_k2_words = [item['word'] for item in sup_items_k2]
                sup_embs_k2 = extract_embeddings(base_model, X_sup_k2)
                p1_proto2, p3_proto2 = evaluate_prototypes(sup_embs_k2, labels_sup_k2_words, test_embs, labels_test_words)
                method_preds_top1['prototype_k2'].extend(p1_proto2)
                method_preds_top3['prototype_k2'].extend(p3_proto2)

            # 5. Adapter Tuning (k=1 shot/class, 40 epochs)
            adapter_net = personalize(base_state, X_sup_k1, Y_sup_k1, 'adapter', vocab, blank, epochs=40, lr=1e-4)
            p1_adapt, p3_adapt = evaluate_digits_ctc(adapter_net, X_test, Y_test, vocab, blank)
            method_preds_top1['adapter_k1'].extend(p1_adapt)
            method_preds_top3['adapter_k1'].extend(p3_adapt)
            del adapter_net

            # 6. Full Fine-Tuning (k=1 shot/class, 40 epochs)
            full_net = personalize(base_state, X_sup_k1, Y_sup_k1, 'full', vocab, blank, epochs=40, lr=1e-4)
            p1_full, p3_full = evaluate_digits_ctc(full_net, X_test, Y_test, vocab, blank)
            method_preds_top1['full_k1'].extend(p1_full)
            method_preds_top3['full_k1'].extend(p3_full)
            del full_net, base_model
            torch.cuda.empty_cache()

        # Compute summary metrics over all pooled test clips
        total_test_clips = sum(len(spk_digit_blocks[b]) for b in blocks)
        spk_res = {}
        for m in methods:
            acc_m, low, high = bootstrap_ci(np.array(method_preds_top1[m]))
            top3_m = float(np.mean(method_preds_top3[m]) * 100.0)
            spk_res[m] = {
                'top1_acc': acc_m,
                'top1_ci95': [low, high],
                'top3_acc': top3_m,
                'n_eval': total_test_clips
            }
            print(f"  Method: {m:<15} | Top-1: {acc_m:5.1f}% [{low:4.1f}%, {high:4.1f}%] | Top-3: {top3_m:5.1f}%")

        spk_res['chance_level'] = 10.0
        spk_res['majority_class'] = 10.0
        spk_res['n_total_clips'] = total_test_clips
        results_digits[spk] = spk_res

    return results_digits


def run_155_words_classification(ua, base_state, vocab):
    """
    Extended Label Set: 155 repeated words across blocks in UASpeech
    (10 digits + 26 letters + 19 commands + 100 common words).
    Evaluated with Prototype Classifier under LOBO folds.
    """
    print("\n" + "=" * 70)
    print("3. EXTENDED 155-WORD VOCABULARY EVALUATION (PROTOTYPE CLASSIFIER)")
    print("=" * 70)

    base_model = LipNet(len(vocab)).to(device)
    base_model.load_state_dict(base_state)
    base_model.eval()

    results_155 = {}

    for spk, blocks in [('F05', ['B1', 'B2', 'B3']), ('F02', ['B1', 'B2'])]:
        desc = "Mild dysarthria" if spk == 'F05' else "Severe dysarthria"
        print(f"\n>>> SPEAKER: {spk} ({desc}) — 155 Repeated Words — LOBO Folds: {blocks}")

        # Group 155 words per block
        spk_155_blocks = {}
        for b in blocks:
            spk_155_blocks[b] = []
            for utt, clip in ua[spk]:
                parts = utt.split('_')
                if parts[1] == b:
                    code = parts[-1].upper()
                    # Include D*, L*, C*, CW* (exclude UW* which are block-specific)
                    if not code.startswith('UW'):
                        spk_155_blocks[b].append({
                            'utt': utt, 'block': b, 'code': code, 'clip': clip
                        })

        for b in blocks:
            print(f"  Block {b}: {len(spk_155_blocks[b])} clips")

        preds_top1_k1 = []
        preds_top3_k1 = []
        preds_top1_k2 = []
        preds_top3_k2 = []

        # LOBO folds
        for fold_idx, test_b in enumerate(blocks):
            support_blocks = [b for b in blocks if b != test_b]
            test_items = spk_155_blocks[test_b]
            X_test = torch.from_numpy(np.stack([item['clip'] for item in test_items]))
            labels_test = [item['code'] for item in test_items]
            test_embs = extract_embeddings(base_model, X_test)

            # k=1 shot/class (155 support clips)
            sup_items_k1 = spk_155_blocks[support_blocks[0]]
            X_sup_k1 = torch.from_numpy(np.stack([item['clip'] for item in sup_items_k1]))
            labels_sup_k1 = [item['code'] for item in sup_items_k1]
            sup_embs_k1 = extract_embeddings(base_model, X_sup_k1)

            p1_k1, p3_k1 = evaluate_prototypes(sup_embs_k1, labels_sup_k1, test_embs, labels_test)
            preds_top1_k1.extend(p1_k1)
            preds_top3_k1.extend(p3_k1)

            # k=2 shots/class (310 support clips, F05 only)
            if spk == 'F05':
                sup_items_k2 = spk_155_blocks[support_blocks[0]] + spk_155_blocks[support_blocks[1]]
                X_sup_k2 = torch.from_numpy(np.stack([item['clip'] for item in sup_items_k2]))
                labels_sup_k2 = [item['code'] for item in sup_items_k2]
                sup_embs_k2 = extract_embeddings(base_model, X_sup_k2)

                p1_k2, p3_k2 = evaluate_prototypes(sup_embs_k2, labels_sup_k2, test_embs, labels_test)
                preds_top1_k2.extend(p1_k2)
                preds_top3_k2.extend(p3_k2)

        # Metrics for k=1
        mean_top1, low, high = bootstrap_ci(np.array(preds_top1_k1))
        mean_top3 = float(np.mean(preds_top3_k1) * 100.0)
        chance = 100.0 / 155.0  # 0.65%

        res_entry = {
            'n_classes': 155,
            'n_eval': len(preds_top1_k1),
            'chance_level': chance,
            'majority_class': chance,
            'prototype_k1': {
                'top1_acc': mean_top1,
                'top1_ci95': [low, high],
                'top3_acc': mean_top3
            }
        }
        print(f"  [155-Class] k=1 shot/class | Top-1: {mean_top1:5.2f}% [{low:4.2f}%, {high:4.2f}%] | Top-3: {mean_top3:5.2f}% | Chance: {chance:.2f}%")

        if spk == 'F05':
            mean_top1_k2, low_k2, high_k2 = bootstrap_ci(np.array(preds_top1_k2))
            mean_top3_k2 = float(np.mean(preds_top3_k2) * 100.0)
            res_entry['prototype_k2'] = {
                'top1_acc': mean_top1_k2,
                'top1_ci95': [low_k2, high_k2],
                'top3_acc': mean_top3_k2
            }
            print(f"  [155-Class] k=2 shots/class| Top-1: {mean_top1_k2:5.2f}% [{low_k2:4.2f}%, {high_k2:4.2f}%] | Top-3: {mean_top3_k2:5.2f}% | Chance: {chance:.2f}%")

        results_155[spk] = res_entry

    del base_model
    torch.cuda.empty_cache()
    return results_155


def main():
    commit = get_git_commit()
    print(f"Git HEAD commit: {commit}")

    # Load base model checkpoint
    ck = torch.load(os.path.join(ROOT, 'source_model.pt'), map_location=device, weights_only=False)
    vocab = list(ck['vocab'])
    blank = len(vocab)
    base_state = ck['model']

    # Load UASpeech dataset
    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()

    # 1. Control Test
    ctrl_model = LipNet(len(vocab)).to(device)
    ctrl_model.load_state_dict(base_state)
    ctrl_res = run_control_test(ctrl_model, vocab, blank)

    # 2. LOBO Digits (Item 2, 3, 4)
    lobo_digits_res = run_lobo_digit_experiments(ua, base_state, vocab, blank)

    # 3. Extended 155 Words (Item 5)
    lobo_155_res = run_155_words_classification(ua, base_state, vocab)

    # Compile Final Phase A2 Results
    phase_a2_output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'git_commit': commit,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
            'phase': 'Phase A2: Diagnosis and Evaluation Redesign'
        },
        'control_test': ctrl_res,
        'lobo_digit_benchmark': lobo_digits_res,
        'lobo_155_words_benchmark': lobo_155_res
    }

    # Save to results/phase_a2_results.json
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'phase_a2_results.json')
    with open(out_path, 'w') as f:
        json.dump(phase_a2_output, f, indent=2)
    print(f"\n[OK] Saved complete Phase A2 results to: {out_path}")


if __name__ == '__main__':
    main()
