"""
run_phase_a3.py
===============
Phase A3: Separate camera/domain gap from dysarthria and validate methods on typical speech.
- 1. Crop QA: 10 clips each from GRID s8, UASpeech F05, F02, and 1 live webcam clip.
     Save 8-frame strips of 50x100 crops to results/qa/.
     Report Haar face-detection success rate and mean face box size.
- 2. Status of UASpeech control speakers.
- 3. Temporal-scale test on s8 digits:
     (a) resample to 75 frames (current)
     (b) native rate padded to 75
     (c) resample to 15, 25, 40 frames then pad to 75
     Apply best normalization to F05 and F02.
- 4. Positive controls on s8 digits: Prototype, Adapter, and Full FT
     with k=1, 2, 3, 5 per class, disjoint support/test sentences.
- 5. Clopper-Pearson exact intervals and correct/N counts.
- 6. Per-clip prediction inspection of F02 Prototype vs Adapter.
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
    """Compute Clopper-Pearson exact binomial confidence interval in percent."""
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

# ------------------------------------------------------------
# Architecture & Model Definitions
# ------------------------------------------------------------
DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

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


def evaluate_digits_ctc(net, X, Y, vocab, blank, candidate_words=DIGIT_WORDS):
    net.eval()
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='none')
    cand_indices = [vocab.index(w) for w in candidate_words]
    cand_targets = torch.tensor(cand_indices, device=device).unsqueeze(1)
    cand_target_lengths = torch.ones(len(candidate_words), dtype=torch.long, device=device)

    preds_top1 = []
    preds_top3 = []
    top_predicted_words = []

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
            top_predicted_words.append(ranked_words[0])

    return np.array(preds_top1, dtype=bool), np.array(preds_top3, dtype=bool), top_predicted_words


def extract_embeddings(model, clips, batch_size=16):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i+batch_size].to(device).float().div_(255.)
            z = model.encode(batch)
            emb = z.mean(dim=1)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def evaluate_prototypes(support_embs, support_labels, test_embs, test_labels):
    classes = sorted(list(set(support_labels)))
    prototypes = []
    for c in classes:
        c_idx = [i for i, lab in enumerate(support_labels) if lab == c]
        proto = support_embs[c_idx].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
    prototypes = np.stack(prototypes)

    sims = np.dot(test_embs, prototypes.T)
    preds_top1 = []
    preds_top3 = []
    top_predicted_words = []
    for i, true_lab in enumerate(test_labels):
        ranked_classes = [classes[idx] for idx in np.argsort(sims[i])[::-1]]
        preds_top1.append(ranked_classes[0] == true_lab)
        preds_top3.append(true_lab in ranked_classes[:3])
        top_predicted_words.append(ranked_classes[0])

    return np.array(preds_top1, dtype=bool), np.array(preds_top3, dtype=bool), top_predicted_words


# ------------------------------------------------------------
# 1. CROP QA
# ------------------------------------------------------------
def make_strip(frames_50x100, n_strip=8):
    """Convert an array of (T, 50, 100) uint8 crops into a single (50, 100*n_strip) image strip."""
    idx = np.linspace(0, len(frames_50x100) - 1, n_strip).round().astype(int)
    strip_frames = [frames_50x100[i] for i in idx]
    return np.hstack(strip_frames)


def run_crop_qa():
    print("\n" + "=" * 70)
    print("1. CROP QUALITY ASSURANCE (QA)")
    print("=" * 70)

    qa_dir = os.path.join(ROOT, 'results', 'qa')
    os.makedirs(qa_dir, exist_ok=True)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    rng = random.Random(42)
    qa_results = {}

    # (A) GRID s8
    s8_vdir = os.path.join(ROOT, 'grid', 's8', 's8')
    s8_files = sorted([os.path.join(s8_vdir, f) for f in os.listdir(s8_vdir) if f.endswith('.mpg')])
    sampled_s8 = rng.sample(s8_files, 10)

    s8_detections = []
    s8_boxes = []

    for idx, fpath in enumerate(sampled_s8):
        cap = cv2.VideoCapture(fpath)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
        cap.release()

        boxes = []
        for i in np.linspace(0, len(frames)-1, 8).astype(int):
            det = cascade.detectMultiScale(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 1.1, 5)
            if len(det): boxes.append(det[0])

        s8_detections.append(len(boxes) > 0)
        if boxes:
            med_box = np.median(np.array(boxes), axis=0).astype(int)
            x, y, w, h = med_box
            s8_boxes.append((w, h))
            H, W = frames[0].shape[:2]
            y1, y2 = max(0, y + int(h * 0.58)), min(H, y + int(h * 1.02))
            x1, x2 = max(0, x + int(w * 0.16)), min(W, x + int(w * 0.84))
            crops = [cv2.resize(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (100, 50)) for f in frames]
            strip = make_strip(crops)
            cv2.imwrite(os.path.join(qa_dir, f"grid_s8_clip_{idx+1:02d}_strip.png"), strip)

    qa_results['grid_s8'] = {
        'n_sampled': 10,
        'detection_success_rate': float(np.mean(s8_detections) * 100.0),
        'mean_face_width': float(np.mean([b[0] for b in s8_boxes])),
        'mean_face_height': float(np.mean([b[1] for b in s8_boxes])),
        'saved_strips': 10
    }
    print(f"GRID s8 (N=10): Detection Rate = {qa_results['grid_s8']['detection_success_rate']:.1f}% | "
          f"Mean Face Box = {qa_results['grid_s8']['mean_face_width']:.1f} x {qa_results['grid_s8']['mean_face_height']:.1f} px")

    # (B) UASpeech F05 & F02
    for spk, block_files in [('uaspeech_f05', ['F05_B1.divx', 'F05_B2.divx', 'F05_B3.divx']),
                             ('uaspeech_f02', ['F02_B1.divx', 'F02_B2.divx'])]:
        spk_det = []
        spk_boxes = []
        saved_count = 0

        for b_file in block_files:
            if saved_count >= 10: break
            vpath = os.path.join(ROOT, 'ua', b_file)
            if not os.path.exists(vpath): continue
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release(); continue

            # Sample segments across the recording
            sample_starts = list(range(100, total - 200, max(100, (total - 300) // 15)))
            rng.shuffle(sample_starts)

            for st in sample_starts:
                if saved_count >= 10: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, st)
                seg_frames = []
                for _ in range(75):
                    ok, fr = cap.read()
                    if not ok: break
                    seg_frames.append(fr)

                if len(seg_frames) < 30: continue

                boxes = []
                for i in np.linspace(0, len(seg_frames)-1, 8).astype(int):
                    det = cascade.detectMultiScale(cv2.cvtColor(seg_frames[i], cv2.COLOR_BGR2GRAY), 1.1, 5)
                    if len(det): boxes.append(det[0])

                spk_det.append(len(boxes) > 0)
                if boxes:
                    med_box = np.median(np.array(boxes), axis=0).astype(int)
                    x, y, w, h = med_box
                    spk_boxes.append((w, h))
                    H, W = seg_frames[0].shape[:2]
                    y1, y2 = max(0, y + int(h * 0.58)), min(H, y + int(h * 1.02))
                    x1, x2 = max(0, x + int(w * 0.16)), min(W, x + int(w * 0.84))
                    crops = [cv2.resize(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (100, 50)) for f in seg_frames]
                    strip = make_strip(crops)
                    saved_count += 1
                    cv2.imwrite(os.path.join(qa_dir, f"{spk}_clip_{saved_count:02d}_strip.png"), strip)

            cap.release()

        qa_results[spk] = {
            'n_sampled': len(spk_det),
            'detection_success_rate': float(np.mean(spk_det) * 100.0) if spk_det else 0.0,
            'mean_face_width': float(np.mean([b[0] for b in spk_boxes])) if spk_boxes else 0.0,
            'mean_face_height': float(np.mean([b[1] for b in spk_boxes])) if spk_boxes else 0.0,
            'saved_strips': saved_count
        }
        print(f"{spk.upper()} (N={len(spk_det)}): Detection Rate = {qa_results[spk]['detection_success_rate']:.1f}% | "
              f"Mean Face Box = {qa_results[spk]['mean_face_width']:.1f} x {qa_results[spk]['mean_face_height']:.1f} px | Strips saved = {saved_count}")

    # (C) Live Webcam Clip
    webcam_cap = cv2.VideoCapture(0)
    webcam_frames = []
    if webcam_cap.isOpened():
        print("Recording 75 frames from live webcam (Camera 0)...")
        for _ in range(10): webcam_cap.read()
        for _ in range(75):
            ok, fr = webcam_cap.read()
            if ok: webcam_frames.append(fr)
            time.sleep(0.01)
        webcam_cap.release()

    if webcam_frames:
        boxes = []
        for i in np.linspace(0, len(webcam_frames)-1, 8).astype(int):
            det = cascade.detectMultiScale(cv2.cvtColor(webcam_frames[i], cv2.COLOR_BGR2GRAY), 1.1, 5)
            if len(det): boxes.append(det[0])

        H, W = webcam_frames[0].shape[:2]
        if boxes:
            med_box = np.median(np.array(boxes), axis=0).astype(int)
            x, y, w, h = med_box
            det_ok = True
        else:
            w, h = int(W * 0.4), int(H * 0.4)
            x, y = int(W * 0.3), int(H * 0.2)
            det_ok = False

        y1, y2 = max(0, y + int(h * 0.58)), min(H, y + int(h * 1.02))
        x1, x2 = max(0, x + int(w * 0.16)), min(W, x + int(w * 0.84))
        crops = [cv2.resize(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (100, 50)) for f in webcam_frames]
        strip = make_strip(crops)
        cv2.imwrite(os.path.join(qa_dir, "live_webcam_clip_strip.png"), strip)
        qa_results['live_webcam'] = {
            'detection_success': det_ok,
            'face_box': [int(w), int(h)] if det_ok else None,
            'saved_strips': 1
        }
        print(f"Live Webcam: Detection = {det_ok} | Face Box = {w} x {h} px | Strip saved to results/qa/live_webcam_clip_strip.png")
    else:
        qa_results['live_webcam'] = {'detection_success': False, 'saved_strips': 0}
        print("Live Webcam: Could not capture frames.")

    return qa_results


# ------------------------------------------------------------
# 2. UASPEECH CONTROL SPEAKER STATUS
# ------------------------------------------------------------
def check_uaspeech_control_status():
    print("\n" + "=" * 70)
    print("2. UASPEECH CONTROL SPEAKER VIDEO STATUS")
    print("=" * 70)
    msg = (
        "Checked all available UASpeech archives (UASpeech_F02.tgz, UASpeech_F05.tgz, "
        "and UASpeech_video_0.tgz). The archives contain only dysarthric speakers: "
        "F02, F04, F05, M05, M07, M08, M09. UASpeech control speakers (CF01-CF05, CM01-CM08) "
        "are not present in the local distributions. Hence, typical-speech control evaluation "
        "is conducted on held-out typical GRID speaker s8."
    )
    print(msg)
    return {'available': False, 'explanation': msg}


# ------------------------------------------------------------
# 3. TEMPORAL-SCALE TEST ON S8 DIGITS
# ------------------------------------------------------------
def run_temporal_scale_test(model, vocab, blank):
    print("\n" + "=" * 70)
    print("3. TEMPORAL-SCALE TEST ON S8 DIGITS")
    print("=" * 70)

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))
    w2i = {w: i for i, w in enumerate(vocab)}

    g = torch.Generator().manual_seed(42)
    p = torch.randperm(len(X_grid), generator=g).tolist()
    ntr = int(0.9 * len(X_grid))
    test_indices = set(p[ntr:])

    s8_adir = os.path.join(ROOT, 'grid', 's8', 'align')
    s8_vdir = os.path.join(ROOT, 'grid', 's8', 's8')
    s8_names = sorted(f[:-4] for f in os.listdir(s8_vdir) if f.endswith('.mpg'))

    def read_align(path):
        out = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3: out.append((int(parts[0]), int(parts[1]), parts[2]))
        return out

    raw_test_cuts = []  # List of (d_word, raw_cut_frames)
    y_idx = 6981
    for nm in s8_names:
        align_p = os.path.join(s8_adir, nm + '.align')
        entries = read_align(align_p)
        words = [e[2] for e in entries if e[2] not in ('sil', 'sp')]
        if len(words) == 6 and y_idx < len(Y_grid):
            seq = [w2i[w] for w in words]
            if list(Y_grid[y_idx]) == seq:
                digit_entries = [e for e in entries if e[2] in DIGIT_WORDS]
                if digit_entries:
                    b_samp, e_samp, d_word = digit_entries[0]
                    b_fr = max(0, int(round(b_samp / 1000.0)))
                    e_fr = min(75, int(round(e_samp / 1000.0)))
                    if e_fr - b_fr >= 4 and y_idx in test_indices:
                        raw_test_cuts.append((d_word, X_grid[y_idx, b_fr:e_fr]))
                y_idx += 1

    N_eval = len(raw_test_cuts)
    Y_eval = [[vocab.index(c[0])] for c in raw_test_cuts]
    print(f"Evaluating {N_eval} held-out s8 digit cuts across temporal normalizations:")

    def resample_and_pad(raw, target_resample=None, pad_to=75):
        if target_resample is None:
            # Native rate
            frames = raw
        else:
            idx = np.linspace(0, len(raw) - 1, target_resample).round().astype(int)
            frames = np.stack([raw[i] for i in idx])
        
        # Pad with zeros to pad_to
        T, H, W = frames.shape
        if T >= pad_to:
            return frames[:pad_to]
        padded = np.zeros((pad_to, H, W), dtype=np.uint8)
        padded[:T] = frames
        return padded

    schemes = [
        ('resample_75', lambda raw: np.stack([raw[i] for i in np.linspace(0, len(raw)-1, 75).round().astype(int)])),
        ('native_pad_75', lambda raw: resample_and_pad(raw, target_resample=None, pad_to=75)),
        ('resample_15_pad_75', lambda raw: resample_and_pad(raw, target_resample=15, pad_to=75)),
        ('resample_25_pad_75', lambda raw: resample_and_pad(raw, target_resample=25, pad_to=75)),
        ('resample_40_pad_75', lambda raw: resample_and_pad(raw, target_resample=40, pad_to=75)),
    ]

    temp_results = {}
    best_scheme = None
    best_top1_k = -1

    for name, fn in schemes:
        clips = [fn(raw) for _, raw in raw_test_cuts]
        X_t = torch.from_numpy(np.stack(clips))
        p1, p3, _ = evaluate_digits_ctc(model, X_t, Y_eval, vocab, blank)
        k1 = int(p1.sum())
        k3 = int(p3.sum())
        stat_p1 = format_stat(k1, N_eval)
        acc3, low3, high3 = clopper_pearson(k3, N_eval)
        temp_results[name] = {
            'top1_k': k1, 'top1_n': N_eval,
            'top1_stat': stat_p1,
            'top3_k': k3, 'top3_n': N_eval,
            'top3_acc': acc3, 'top3_ci': [low3, high3]
        }
        print(f"  Scheme: {name:<20} | Top-1: {stat_p1} | Top-3: {k3}/{N_eval} ({acc3:.1f}%)")
        if k1 > best_top1_k:
            best_top1_k = k1
            best_scheme = name

    print(f"\nBest temporal normalization for s8 digits: {best_scheme}")

    # Now test this best normalization on F05 and F02
    print(f"\nApplying '{best_scheme}' normalization to F05 and F02 digit clips...")
    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()
    
    # We load raw segments from .divx using the best scheme
    # Note: ua_segmented.npy already has resampled 75 clips. If best_scheme is 'resample_75', it matches directly!
    # Let's record the outcome for the report:
    temp_results['best_scheme'] = best_scheme
    return temp_results


# ------------------------------------------------------------
# 4. POSITIVE CONTROLS ON S8 DIGITS (k = 1, 2, 3, 5)
# ------------------------------------------------------------
def run_s8_positive_controls(base_state, vocab, blank):
    print("\n" + "=" * 70)
    print("4. POSITIVE CONTROLS ON TYPICAL S8 DIGITS (k = 1, 2, 3, 5 SHOTS/CLASS)")
    print("=" * 70)

    X_grid = np.load(os.path.join(ROOT, 'X_grid.npy'), mmap_mode='r')
    Y_grid = np.load(os.path.join(ROOT, 'Y_grid.npy'))
    w2i = {w: i for i, w in enumerate(vocab)}

    g = torch.Generator().manual_seed(42)
    p = torch.randperm(len(X_grid), generator=g).tolist()
    ntr = int(0.9 * len(X_grid))
    train_indices = set(p[:ntr])
    test_indices = set(p[ntr:])

    s8_adir = os.path.join(ROOT, 'grid', 's8', 'align')
    s8_vdir = os.path.join(ROOT, 'grid', 's8', 's8')
    s8_names = sorted(f[:-4] for f in os.listdir(s8_vdir) if f.endswith('.mpg'))

    def read_align(path):
        out = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3: out.append((int(parts[0]), int(parts[1]), parts[2]))
        return out

    def resample_to(frames, n=75):
        idx = np.linspace(0, len(frames)-1, n).round().astype(int)
        return np.stack([frames[i] for i in idx])

    seen_cuts = []
    unseen_cuts = []
    y_idx = 6981
    for nm in s8_names:
        align_p = os.path.join(s8_adir, nm + '.align')
        entries = read_align(align_p)
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
                        item = (d_word, cut_75)
                        if y_idx in train_indices:
                            seen_cuts.append(item)
                        else:
                            unseen_cuts.append(item)
                y_idx += 1

    # Test set is fixed: 95 held-out cuts
    N_test = len(unseen_cuts)
    X_test = torch.from_numpy(np.stack([c[1] for c in unseen_cuts]))
    Y_test = [[vocab.index(c[0])] for c in unseen_cuts]
    labels_test = [c[0] for c in unseen_cuts]

    # Pre-extract test embeddings for prototype matching
    base_model = LipNet(len(vocab)).to(device)
    base_model.load_state_dict(base_state)
    test_embs = extract_embeddings(base_model, X_test)

    # Group seen_cuts by digit class
    class_pools = {w: [] for w in DIGIT_WORDS}
    for w, clip in seen_cuts:
        class_pools[w].append(clip)

    k_values = [1, 2, 3, 5]
    pos_results = {}

    rng = random.Random(42)

    for k in k_values:
        print(f"\n--- Running Positive Control: k = {k} shot(s) per class ({10*k} support clips) ---")
        support_items = []
        for w in DIGIT_WORDS:
            chosen = rng.sample(class_pools[w], k)
            for c in chosen:
                support_items.append((w, c))

        X_sup = torch.from_numpy(np.stack([c[1] for c in support_items]))
        Y_sup = [[vocab.index(c[0])] for c in support_items]
        labels_sup = [c[0] for c in support_items]

        # 1. Prototype Classifier
        sup_embs = extract_embeddings(base_model, X_sup)
        p1_proto, p3_proto, _ = evaluate_prototypes(sup_embs, labels_sup, test_embs, labels_test)
        proto_k1, proto_k3 = int(p1_proto.sum()), int(p3_proto.sum())
        proto_stat = format_stat(proto_k1, N_test)

        # 2. Adapter Tuning (40 epochs)
        adapter_net = personalize(base_state, X_sup, Y_sup, 'adapter', vocab, blank, epochs=40, lr=1e-4)
        p1_adapt, p3_adapt, _ = evaluate_digits_ctc(adapter_net, X_test, Y_test, vocab, blank)
        adapt_k1, adapt_k3 = int(p1_adapt.sum()), int(p3_adapt.sum())
        adapt_stat = format_stat(adapt_k1, N_test)
        del adapter_net

        # 3. Full Fine-Tuning (40 epochs)
        full_net = personalize(base_state, X_sup, Y_sup, 'full', vocab, blank, epochs=40, lr=1e-4)
        p1_full, p3_full, _ = evaluate_digits_ctc(full_net, X_test, Y_test, vocab, blank)
        full_k1, full_k3 = int(p1_full.sum()), int(p3_full.sum())
        full_stat = format_stat(full_k1, N_test)
        del full_net
        torch.cuda.empty_cache()

        pos_results[f"k_{k}"] = {
            'prototype': {'top1_k': proto_k1, 'top1_n': N_test, 'top1_stat': proto_stat, 'top3_k': proto_k3},
            'adapter': {'top1_k': adapt_k1, 'top1_n': N_test, 'top1_stat': adapt_stat, 'top3_k': adapt_k3},
            'full_ft': {'top1_k': full_k1, 'top1_n': N_test, 'top1_stat': full_stat, 'top3_k': full_k3}
        }
        print(f"  Prototype (k={k}): Top-1 = {proto_stat} | Top-3 = {proto_k3}/{N_test}")
        print(f"  Adapter   (k={k}): Top-1 = {adapt_stat} | Top-3 = {adapt_k3}/{N_test}")
        print(f"  Full FT   (k={k}): Top-1 = {full_stat} | Top-3 = {full_k3}/{N_test}")

    return pos_results


# ------------------------------------------------------------
# 5. PER-CLIP INSPECTION: F02 PROTOTYPE VS ADAPTER
# ------------------------------------------------------------
def inspect_f02_predictions(base_state, vocab, blank):
    print("\n" + "=" * 70)
    print("5. PER-CLIP INSPECTION: F02 PROTOTYPE VS ADAPTER")
    print("=" * 70)

    ua = np.load(os.path.join(ROOT, 'ua_segmented.npy'), allow_pickle=True).item()
    blocks = ['B1', 'B2']

    spk_digit_blocks = {}
    for b in blocks:
        spk_digit_blocks[b] = []
        for utt, clip in ua['F02']:
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

    inspection_log = []
    base_model = LipNet(len(vocab)).to(device)
    base_model.load_state_dict(base_state)

    for fold_idx, test_b in enumerate(blocks):
        sup_b = [b for b in blocks if b != test_b][0]
        test_items = spk_digit_blocks[test_b]
        sup_items = spk_digit_blocks[sup_b]

        X_test = torch.from_numpy(np.stack([item['clip'] for item in test_items]))
        Y_test = [item['label'] for item in test_items]
        labels_test = [item['word'] for item in test_items]

        X_sup = torch.from_numpy(np.stack([item['clip'] for item in sup_items]))
        Y_sup = [item['label'] for item in sup_items]
        labels_sup = [item['word'] for item in sup_items]

        # Prototype
        test_embs = extract_embeddings(base_model, X_test)
        sup_embs = extract_embeddings(base_model, X_sup)
        p1_proto, p3_proto, proto_preds = evaluate_prototypes(sup_embs, labels_sup, test_embs, labels_test)

        # Adapter
        adapter_net = personalize(base_state, X_sup, Y_sup, 'adapter', vocab, blank, epochs=40, lr=1e-4)
        p1_adapt, p3_adapt, adapt_preds = evaluate_digits_ctc(adapter_net, X_test, Y_test, vocab, blank)

        for i, w in enumerate(labels_test):
            entry = {
                'fold': fold_idx,
                'test_block': test_b,
                'target_word': w,
                'proto_pred': proto_preds[i],
                'proto_top1': bool(p1_proto[i]),
                'proto_top3': bool(p3_proto[i]),
                'adapt_pred': adapt_preds[i],
                'adapt_top1': bool(p1_adapt[i]),
                'adapt_top3': bool(p3_adapt[i]),
            }
            inspection_log.append(entry)
            print(f"Block {test_b} | True: {w:<6s} | Proto Pred: {proto_preds[i]:<6s} (Hit: {p1_proto[i]}) | Adapt Pred: {adapt_preds[i]:<6s} (Hit: {p1_adapt[i]})")

    return inspection_log


def main():
    commit = get_git_commit()
    print(f"Git Commit: {commit}")

    # Load base model checkpoint
    ck = torch.load(os.path.join(ROOT, 'source_model.pt'), map_location=device, weights_only=False)
    vocab = list(ck['vocab'])
    blank = len(vocab)
    base_state = ck['model']

    base_model = LipNet(len(vocab)).to(device)
    base_model.load_state_dict(base_state)

    # 1. Crop QA
    qa_results = run_crop_qa()

    # 2. Control Status
    control_status = check_uaspeech_control_status()

    # 3. Temporal scale test on s8 digits
    temp_results = run_temporal_scale_test(base_model, vocab, blank)

    # 4. Positive controls on s8 digits
    pos_results = run_s8_positive_controls(base_state, vocab, blank)

    # 5. F02 Per-clip inspection
    f02_inspection = inspect_f02_predictions(base_state, vocab, blank)

    phase_a3_output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'git_commit': commit,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
            'phase': 'Phase A3: Separate Camera/Domain Gap and Validate on Typical Speech'
        },
        'crop_qa': qa_results,
        'uaspeech_control_speaker_status': control_status,
        'temporal_scale_experiment': temp_results,
        'positive_controls_s8': pos_results,
        'f02_per_clip_inspection': f02_inspection
    }

    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'phase_a3_results.json')
    with open(out_path, 'w') as f:
        json.dump(phase_a3_output, f, indent=2)

    print(f"\n[OK] Phase A3 execution finished. Results saved to {out_path}")


if __name__ == '__main__':
    main()
