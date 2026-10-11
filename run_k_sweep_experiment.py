"""
run_k_sweep_experiment.py
=========================
Rigorous Few-Shot Personalization Evaluation on UASpeech (F05 & F02).
- Zero-leakage disjoint block splits:
    F05: Held-out test block B3 (10 clips, 0-9), support pool B1 & B2 (20 clips).
    F02: Held-out test block B2 (10 clips, 0-9), support pool B1 (10 clips).
- Closed-set single-word constrained decoding + Top-1 / Top-3 accuracy.
- Unconstrained CTC sentence decode (reporting WER/CER artifact).
- 5 random seeds (0, 1, 2, 3, 4) for k in {1, 3, 5, 10, 20}.
- Saves full metrics (mean ± std) to results.json.
- Regenerates fig_data_efficiency.png.
"""

import os, sys, copy, json, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ------------------------------------------------------------
# 1. Model & Adaptation Classes (Matched to Repository Architecture)
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


# ------------------------------------------------------------
# 2. Decoding & Metrics Helpers
# ------------------------------------------------------------
def ctc_decode(logits, blank, vocab):
    ids, out, prev = logits.argmax(-1).tolist(), [], None
    for k in ids:
        if k != prev and k != blank: out.append(vocab[k])
        prev = k
    return out


def levenshtein(a, b):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1): dp[i, 0] = i
    for j in range(len(b) + 1): dp[0, j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1,
                           dp[i-1, j-1] + (0 if a[i-1] == b[j-1] else 1))
    return dp[len(a), len(b)]


def wer(ref, hyp):
    return levenshtein(ref, hyp) / max(1, len(ref))


def cer(ref, hyp):
    r, h = " ".join(ref), " ".join(hyp)
    return levenshtein(r, h) / max(1, len(r))


DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def evaluate_isolated_words(net, X, Y, vocab, blank, candidate_words=DIGIT_WORDS):
    """
    Evaluates isolated word clips:
    - Constrained closed-set classification via exact CTC sequence scoring
    - Top-1 Word Accuracy & Top-3 Accuracy
    - Constrained WER (bounded in [0%, 100%])
    - Unconstrained Greedy CTC sentence decode (raw sentence WER & CER)
    """
    net.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='none')
    cand_indices = [vocab.index(w) for w in candidate_words]
    cand_targets = torch.tensor(cand_indices, device=device).unsqueeze(1)
    cand_target_lengths = torch.ones(len(candidate_words), dtype=torch.long, device=device)

    top1_correct = 0
    top3_correct = 0
    raw_wer_sum = 0.0
    raw_cer_sum = 0.0

    with torch.no_grad():
        for i in range(len(X)):
            bx = X[i:i+1].to(device).float().div_(255.)
            logits = net(bx)[0]  # (T=75, V+1)

            # --- Constrained Closed-Set CTC Scoring ---
            log_probs = logits.log_softmax(-1).unsqueeze(1)  # (T, 1, V+1)
            log_probs_cand = log_probs.expand(-1, len(candidate_words), -1)  # (T, N_cand, V+1)
            in_lengths = torch.full((len(candidate_words),), logits.shape[0], dtype=torch.long, device=device)

            losses = ctc_loss_fn(log_probs_cand, cand_targets, in_lengths, cand_target_lengths)
            scores = -losses.cpu().numpy()

            ranked_idx = np.argsort(scores)[::-1]
            ranked_words = [candidate_words[idx] for idx in ranked_idx]

            ref_word = vocab[Y[i][0]]

            if ranked_words[0] == ref_word:
                top1_correct += 1
            if ref_word in ranked_words[:3]:
                top3_correct += 1

            # --- Unconstrained Sentence CTC Decode ---
            hyp = ctc_decode(logits.float().cpu(), blank, vocab)
            ref_list = [ref_word]
            raw_wer_sum += wer(ref_list, hyp)
            raw_cer_sum += cer(ref_list, hyp)

    n = len(X)
    top1_acc = (top1_correct / n) * 100.0
    top3_acc = (top3_correct / n) * 100.0
    constrained_wer = 100.0 - top1_acc
    raw_wer = (raw_wer_sum / n) * 100.0
    raw_cer = (raw_cer_sum / n) * 100.0

    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'constrained_wer': constrained_wer,
        'raw_wer': raw_wer,
        'raw_cer': raw_cer,
        'n': n
    }


def personalize(base_state, Xs, Ys, k, mode, vocab, blank, epochs=40, lr=1e-4):
    """Adapt to speaker using k support examples."""
    base = LipNet(len(vocab)).to(device)
    base.load_state_dict(base_state)
    net = PersonalizedLipNet(base, mode=mode).to(device) if mode != 'full' else base
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    Xk, Yk = Xs[:k], Ys[:k]
    net.train()
    for _ in range(epochs):
        for i in range(0, len(Xk), 2):
            bx = Xk[i:i+2].to(device).float().div_(255.)
            ys = Yk[i:i+2]
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
# 3. Main Experiment Execution
# ------------------------------------------------------------
def main():
    ck = torch.load(os.path.join(ROOT, 'source_model.pt'), map_location=device, weights_only=False)
    vocab = list(ck['vocab'])
    V = len(vocab)
    BLANK = V
    base_state = ck['model']

    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()

    # Partition digit clips into blocks
    def extract_blocks(spk):
        blocks = {}
        for utt, clip in ua[spk]:
            parts = utt.split('_')
            blk = parts[1]
            code = parts[-1].upper()
            if code.startswith('D') and code[1:].isdigit():
                digit_val = int(code[1:])
                word = DIGIT_WORDS[digit_val]
                if blk not in blocks:
                    blocks[blk] = []
                blocks[blk].append({
                    'utt': utt, 'block': blk, 'word': word,
                    'label': [vocab.index(word)], 'clip': clip
                })
        return blocks

    speakers_config = {
        'F05': {
            'test_block': 'B3',
            'support_blocks': ['B1', 'B2'],
            'ks': [0, 1, 3, 5, 10, 20],
            'description': 'mild dysarthria (95% intelligibility)'
        },
        'F02': {
            'test_block': 'B2',
            'support_blocks': ['B1'],
            'ks': [0, 1, 3, 5, 10],
            'description': 'severe dysarthria (29% intelligibility)'
        }
    }

    SEEDS = [0, 1, 2, 3, 4]
    MODES = ['full', 'adapter']
    final_results = {}

    print("\n" + "=" * 70)
    print("STARTING FEW-SHOT PERSONALIZATION EXPERIMENT (5 SEEDS)")
    print("=" * 70)

    for spk, cfg in speakers_config.items():
        print(f"\n>>> SPEAKER: {spk} ({cfg['description']})")
        blocks = extract_blocks(spk)

        # Held-out test set
        test_items = blocks[cfg['test_block']]
        X_test = torch.from_numpy(np.stack([item['clip'] for item in test_items]))
        Y_test = [item['label'] for item in test_items]
        print(f"  Held-out Test Block: {cfg['test_block']} ({len(test_items)} clips, exactly digits 0-9)")

        # Support candidate pool
        pool_items = [item for b in cfg['support_blocks'] for item in blocks[b]]
        print(f"  Disjoint Support Pool: {cfg['support_blocks']} ({len(pool_items)} clips total)")

        final_results[spk] = {}

        for mode in MODES:
            print(f"\n  --- Mode: {mode.upper()} ---")
            mode_results = []

            for k in cfg['ks']:
                if k == 0:
                    # Deterministic Zero-Shot evaluation
                    net = LipNet(V).to(device)
                    net.load_state_dict(base_state)
                    res = evaluate_isolated_words(net, X_test, Y_test, vocab, BLANK)
                    del net

                    entry = {
                        'k': 0,
                        'top1_acc_mean': float(res['top1_acc']),
                        'top1_acc_std': 0.0,
                        'top3_acc_mean': float(res['top3_acc']),
                        'top3_acc_std': 0.0,
                        'constrained_wer_mean': float(res['constrained_wer']),
                        'constrained_wer_std': 0.0,
                        'raw_wer_mean': float(res['raw_wer']),
                        'raw_wer_std': 0.0,
                        'raw_cer_mean': float(res['raw_cer']),
                        'raw_cer_std': 0.0,
                        'n_test': len(X_test),
                        'seeds': [res]
                    }
                    mode_results.append(entry)
                    print(f"  k= 0 | Top-1 Acc: {entry['top1_acc_mean']:5.1f}% | "
                          f"Top-3 Acc: {entry['top3_acc_mean']:5.1f}% | "
                          f"Constrained WER: {entry['constrained_wer_mean']:5.1f}% | "
                          f"Raw CTC WER: {entry['raw_wer_mean']:5.1f}%")
                else:
                    seed_runs = []
                    t0_k = time.time()
                    for seed in SEEDS:
                        rng = np.random.RandomState(seed)
                        sampled_idx = rng.choice(len(pool_items), size=k, replace=False)
                        sampled_items = [pool_items[i] for i in sampled_idx]

                        Xs = torch.from_numpy(np.stack([item['clip'] for item in sampled_items]))
                        Ys = [item['label'] for item in sampled_items]

                        adapted_net = personalize(
                            base_state, Xs, Ys, k, mode, vocab, BLANK,
                            epochs=40, lr=1e-4
                        )

                        eval_res = evaluate_isolated_words(adapted_net, X_test, Y_test, vocab, BLANK)
                        seed_runs.append(eval_res)
                        del adapted_net
                        torch.cuda.empty_cache()

                    # Aggregate over 5 seeds
                    top1s = [r['top1_acc'] for r in seed_runs]
                    top3s = [r['top3_acc'] for r in seed_runs]
                    cwers = [r['constrained_wer'] for r in seed_runs]
                    raw_wers = [r['raw_wer'] for r in seed_runs]
                    raw_cers = [r['raw_cer'] for r in seed_runs]

                    entry = {
                        'k': k,
                        'top1_acc_mean': float(np.mean(top1s)),
                        'top1_acc_std': float(np.std(top1s)),
                        'top3_acc_mean': float(np.mean(top3s)),
                        'top3_acc_std': float(np.std(top3s)),
                        'constrained_wer_mean': float(np.mean(cwers)),
                        'constrained_wer_std': float(np.std(cwers)),
                        'raw_wer_mean': float(np.mean(raw_wers)),
                        'raw_wer_std': float(np.std(raw_wers)),
                        'raw_cer_mean': float(np.mean(raw_cers)),
                        'raw_cer_std': float(np.std(raw_cers)),
                        'n_test': len(X_test),
                        'seeds': seed_runs
                    }
                    mode_results.append(entry)
                    elapsed = time.time() - t0_k
                    print(f"  k={k:2d} | Top-1 Acc: {entry['top1_acc_mean']:5.1f}±{entry['top1_acc_std']:4.1f}% | "
                          f"Top-3 Acc: {entry['top3_acc_mean']:5.1f}±{entry['top3_acc_std']:4.1f}% | "
                          f"Constrained WER: {entry['constrained_wer_mean']:5.1f}±{entry['constrained_wer_std']:4.1f}% | "
                          f"Raw CTC WER: {entry['raw_wer_mean']:5.1f}% ({elapsed:.1f}s)")

            final_results[spk][mode] = mode_results

    # Save to results.json
    out_json = os.path.join(ROOT, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[OK] Saved all experiment results to: {out_json}")

    # Generate Figure
    _plot_paper_figure(final_results)

    # Print Markdown Tables
    _print_markdown_tables(final_results)


def _plot_paper_figure(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Word Accuracy (Top-1 %) vs k
    ax1 = axes[0]
    colors = {'F05': '#1f77b4', 'F02': '#d62728'}
    styles = {'full': '--s', 'adapter': '-o'}

    for spk in ['F05', 'F02']:
        for mode in ['full', 'adapter']:
            pts = res[spk][mode]
            ks = [p['k'] for p in pts]
            means = [p['top1_acc_mean'] for p in pts]
            stds = [p['top1_acc_std'] for p in pts]
            lbl = f"{spk} ({'Mild' if spk=='F05' else 'Severe'}) — {mode.capitalize()}"
            ax1.errorbar(ks, means, yerr=stds, fmt=styles[mode], color=colors[spk],
                         label=lbl, capsize=4, capthick=1.2, alpha=0.9)

    ax1.set_xlabel('Number of Adaptation Examples ($k$)', fontsize=11)
    ax1.set_ylabel('Word Classification Accuracy (%)', fontsize=11)
    ax1.set_title('Top-1 Word Accuracy vs. Data Budget (Mean ± Std over 5 Seeds)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim(-5, 105)
    ax1.legend(loc='lower right', framealpha=0.9)

    # Subplot 2: Constrained WER vs k
    ax2 = axes[1]
    for spk in ['F05', 'F02']:
        for mode in ['full', 'adapter']:
            pts = res[spk][mode]
            ks = [p['k'] for p in pts]
            means = [p['constrained_wer_mean'] for p in pts]
            stds = [p['constrained_wer_std'] for p in pts]
            lbl = f"{spk} — {mode.capitalize()}"
            ax2.errorbar(ks, means, yerr=stds, fmt=styles[mode], color=colors[spk],
                         label=lbl, capsize=4, capthick=1.2, alpha=0.9)

    ax2.set_xlabel('Number of Adaptation Examples ($k$)', fontsize=11)
    ax2.set_ylabel('Constrained Word Error Rate (%)', fontsize=11)
    ax2.set_title('Constrained WER vs. Data Budget (Mean ± Std over 5 Seeds)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    fig_path = os.path.join(ROOT, 'fig_data_efficiency.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Regenerated and saved figure to: {fig_path}")


def _print_markdown_tables(res):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS TABLES (MEAN ± STD OVER 5 RANDOM SEEDS)")
    print("=" * 70)

    for spk in ['F05', 'F02']:
        desc = "Mild dysarthria (95% intelligibility)" if spk == 'F05' else "Severe dysarthria (29% intelligibility)"
        print(f"\n### Speaker {spk} — {desc}")
        print("\n| $k$ (shots) | Full Fine-Tune Top-1 Acc | Adapter Top-1 Acc | Full Constrained WER | Adapter Constrained WER | Full Top-3 Acc | Adapter Top-3 Acc |")
        print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        pts_full = {p['k']: p for p in res[spk]['full']}
        pts_adapt = {p['k']: p for p in res[spk]['adapter']}
        all_ks = sorted(pts_full.keys())

        for k in all_ks:
            f = pts_full[k]
            a = pts_adapt[k]
            if k == 0:
                print(f"| {k:2d} (Zero-shot) | {f['top1_acc_mean']:.1f}% | {a['top1_acc_mean']:.1f}% | "
                      f"{f['constrained_wer_mean']:.1f}% | {a['constrained_wer_mean']:.1f}% | "
                      f"{f['top3_acc_mean']:.1f}% | {a['top3_acc_mean']:.1f}% |")
            else:
                print(f"| {k:2d} | {f['top1_acc_mean']:.1f} ± {f['top1_acc_std']:.1f}% | "
                      f"{a['top1_acc_mean']:.1f} ± {a['top1_acc_std']:.1f}% | "
                      f"{f['constrained_wer_mean']:.1f} ± {f['constrained_wer_std']:.1f}% | "
                      f"{a['constrained_wer_mean']:.1f} ± {a['constrained_wer_std']:.1f}% | "
                      f"{f['top3_acc_mean']:.1f} ± {f['top3_acc_std']:.1f}% | "
                      f"{a['top3_acc_mean']:.1f} ± {a['top3_acc_std']:.1f}% |")


if __name__ == '__main__':
    main()
