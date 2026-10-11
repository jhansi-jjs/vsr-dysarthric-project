"""
LIVE WEBCAM DEMO — Speaker-Adaptive Visual Speech Recognition
================================================================
Two modes:
  [1] Baseline  — record yourself saying a GRID word/sentence,
                  see what the source model (trained on typical
                  speakers) predicts, silently, no audio used.
  [2] Personalize — record k labeled examples of YOUR speech
                  (say them however you actually talk — mumbled,
                  fast, unclear, whatever), fine-tune live on your
                  GPU in a few seconds, then test on a new clip and
                  compare the prediction before vs after adaptation.
                  This recreates your Step 7/8 k-sweep experiment,
                  live, on your own face.

Run from Anaconda Prompt with dl-env active, from your VSR_project
folder (where source_model.pt lives):

    conda activate dl-env
    cd C:\\Users\\jhans\\VSR_project
    python live_demo.py
"""

import os, sys, time, copy
import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = os.getcwd()
CKPT = os.path.join(ROOT, 'source_model.pt')
if not os.path.exists(CKPT):
    print(f"ERROR: {CKPT} not found. Run this script from your VSR_project folder.")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
T_LEN, OUT_H, OUT_W = 75, 50, 100
RECORD_SECONDS = 3


# ------------------------------------------------------------
# Model (identical to notebook Step 3)
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


def ctc_decode(logits, blank, vocab):
    ids, out, prev = logits.argmax(-1).tolist(), [], None
    for k in ids:
        if k != prev and k != blank: out.append(vocab[k])
        prev = k
    return out


def personalize(base_state, Xs, Ys, k, mode, vocab, blank, epochs=60, lr=1e-4):
    base = LipNet(len(vocab)).to(device); base.load_state_dict(base_state)
    net = PersonalizedLipNet(base, mode=mode).to(device) if mode != 'full' else base
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    Xk, Yk = Xs[:k], Ys[:k]
    net.train()
    for _ in range(epochs):
        for i in range(0, len(Xk), 2):
            bx = Xk[i:i + 2].to(device).float().div_(255.)
            ys = Yk[i:i + 2]
            tgt = torch.cat([torch.tensor(t) for t in ys]).to(device)
            tl = torch.tensor([len(t) for t in ys], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            out = net(bx)
            il = torch.full((len(ys),), out.shape[1], dtype=torch.long)
            loss = ctc(out.log_softmax(2).permute(1, 0, 2), tgt, il, tl)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    net.eval()
    return net


# ------------------------------------------------------------
# Resampling, Face detection, Mouth crop & Debug strip
# ------------------------------------------------------------
def resample_to_25fps(raw_frames_with_ts, target_fps=25.0, total_frames=T_LEN):
    """
    Resample captured webcam frames with timestamps to exact 25.0 FPS grid (75 frames),
    matching GRID corpus temporal spacing (40ms per frame).
    """
    if not raw_frames_with_ts:
        return []
    if isinstance(raw_frames_with_ts[0], (tuple, list)) and len(raw_frames_with_ts[0]) == 2:
        times = [t for t, f in raw_frames_with_ts]
        frames = [f for t, f in raw_frames_with_ts]
        t_start = times[0]
        target_times = [t_start + k * (1.0 / target_fps) for k in range(total_frames)]
        resampled = []
        for tt in target_times:
            idx = min(range(len(times)), key=lambda i: abs(times[i] - tt))
            resampled.append(frames[idx])
        return resampled
    else:
        frames = raw_frames_with_ts
        idx = np.linspace(0, len(frames) - 1, total_frames).round().astype(int)
        return [frames[i] for i in idx]


def save_debug_strip(clip, label="Live Clip (CLI)"):
    debug_dir = os.path.join(ROOT, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    filename = f"debug_cli_{int(time.time())}.png"
    filepath = os.path.join(debug_dir, filename)
    rows = []
    for r in range(5):
        row_frames = [clip[r * 15 + c] for c in range(15)]
        rows.append(np.hstack(row_frames))
    grid_img = np.vstack(rows)
    header = np.zeros((30, grid_img.shape[1]), dtype=np.uint8) + 245
    header_bgr = cv2.cvtColor(header, cv2.COLOR_GRAY2BGR)
    grid_bgr = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
    info_text = (f"{label} | 75 frames @ 25 FPS (3.0s) | "
                 f"Mean: {clip.mean():.1f} | Std: {clip.std():.1f}")
    cv2.putText(header_bgr, info_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    combined = np.vstack([header_bgr, grid_bgr])
    cv2.imwrite(filepath, combined)
    print(f"[Debug] Saved 75-frame debug strip to: {filepath}")
    return filepath


def record_and_crop(seconds=RECORD_SECONDS):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return None

    raw_buffer = []
    print(f"\nRecording in 2 seconds... get ready to speak silently.")
    t0 = time.time()
    while time.time() - t0 < 2:
        ok, fr = cap.read()
        if ok:
            cv2.putText(fr, "GET READY...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow('Live VSR Demo', fr)
            cv2.waitKey(1)

    print("RECORDING — speak now (silently)...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, fr = cap.read()
        if not ok: break
        raw_buffer.append((time.perf_counter(), fr))
        disp = fr.copy()
        cv2.putText(disp, "RECORDING...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Live VSR Demo', disp)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if len(raw_buffer) < 10:
        print("Recording too short, try again.")
        return None

    # Resample to exact 25.0 FPS (75 frames at 40ms interval) matching GRID corpus
    frames = resample_to_25fps(raw_buffer, target_fps=25.0, total_frames=T_LEN)

    # One stable face box for the whole clip (identical logic to notebook Steps 2 & 6)
    boxes = []
    for i in np.linspace(0, len(frames) - 1, 8).astype(int):
        det = CASCADE.detectMultiScale(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 1.1, 5)
        if len(det): boxes.append(det[0])
    if not boxes:
        print("No face detected — check lighting/camera position, try again.")
        return None

    x, y, w, h = np.median(np.array(boxes), axis=0).astype(int)
    H, W = frames[0].shape[:2]
    y1, y2 = max(0, y + int(h * 0.58)), min(H, y + int(h * 1.02))
    x1, x2 = max(0, x + int(w * 0.16)), min(W, x + int(w * 0.84))
    if y2 - y1 < 10 or x2 - x1 < 10:
        print("Crop region too small, try again.")
        return None

    roi = [cv2.resize(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (OUT_W, OUT_H)) for f in frames]
    roi = np.array(roi, dtype=np.uint8)

    # Luminance & contrast distribution matching:
    # Match GRID training stats (mean ~148.0, std ~27.0)
    if roi.std() > 1.0:
        cur_m = float(roi.mean())
        cur_s = float(roi.std())
        normalized = (roi.astype(np.float32) - cur_m) * (27.0 / max(cur_s, 1.0)) + 148.0
        roi = np.clip(normalized, 0, 255).astype(np.uint8)

    save_debug_strip(roi)
    return roi


def transcribe(clip, model, vocab, blank):
    model.eval()
    bx = torch.from_numpy(clip).unsqueeze(0).to(device).float().div_(255.)
    with torch.no_grad():
        hyp = ctc_decode(model(bx)[0].float().cpu(), blank, vocab)
    return hyp


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    vocab = ck['vocab']; V = len(vocab); BLANK = V
    base_state = ck['model']
    print(f"\nLoaded source model. Vocab ({V} words): {vocab}\n")

    print("Choose mode:")
    print("  1 = Baseline transcription (say any GRID word/sentence)")
    print("  2 = Live few-shot personalization (record k examples, then test)")
    choice = input("> ").strip()

    if choice == '1':
        model = LipNet(V).to(device); model.load_state_dict(base_state); model.eval()
        while True:
            clip = record_and_crop()
            if clip is not None:
                hyp = transcribe(clip, model, vocab, BLANK)
                print("\nPREDICTED:", ' '.join(hyp) if hyp else '(nothing detected)')
            again = input("\nRecord another? (y/n) > ").strip().lower()
            if again != 'y': break

    elif choice == '2':
        print(f"\nVocab available for labels: {vocab}")
        k = int(input("How many training examples to record (try 10)? > ").strip())
        Xs, Ys = [], []
        for i in range(k):
            print(f"\n--- Training example {i+1}/{k} ---")
            label = input(f"Type the word you're about to say (must be exactly one of the vocab words above): ").strip().lower()
            if label not in vocab:
                print("Not in vocab, skipping this one.")
                continue
            clip = record_and_crop()
            if clip is not None:
                Xs.append(clip); Ys.append([vocab.index(label)])

        if len(Xs) < 2:
            print("Not enough examples collected, exiting.")
            return

        Xs_t = torch.from_numpy(np.stack(Xs))

        print("\nTesting BASELINE (before personalization) on a new clip...")
        input("Press Enter, then say any word from the vocab...")
        test_clip = record_and_crop()
        base_model = LipNet(V).to(device); base_model.load_state_dict(base_state); base_model.eval()
        hyp_before = transcribe(test_clip, base_model, vocab, BLANK)
        print("BEFORE personalization:", ' '.join(hyp_before) if hyp_before else '(nothing detected)')

        print(f"\nFine-tuning on your {len(Xs)} examples (few seconds)...")
        personalized = personalize(base_state, Xs_t, Ys, len(Xs), 'full', vocab, BLANK)
        hyp_after = transcribe(test_clip, personalized, vocab, BLANK)
        print("AFTER  personalization:", ' '.join(hyp_after) if hyp_after else '(nothing detected)')

    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()
