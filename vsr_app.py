"""
VSR Desktop App — Speaker-Adaptive Visual Speech Recognition
=============================================================
Tkinter GUI wrapping the model and recording logic from live_demo.py.

Run from the dl-env conda environment:
    conda activate dl-env
    cd C:\\Users\\jhans\\VSR_project
    python vsr_app.py

Every transcription is a REAL forward pass through the model at the
moment the button is clicked.  Nothing is hardcoded, cached, or faked.
All predictions are logged to demo_log.json with timestamps.
"""

import os, sys, time, copy, json, threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn

from grammar_decoder import grammar_constrained_decode
from phrase_classifier import (
    PhraseEnrollment, classify_prototype,
    train_linear_head, classify_linear
)

# ----------------------------------------------------------------
# Paths & device
# ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, 'source_model.pt')
LOG_PATH = os.path.join(ROOT, 'demo_log.json')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
T_LEN, OUT_H, OUT_W = 75, 50, 100
RECORD_SECONDS = 3
COUNTDOWN_SECONDS = 2


# ----------------------------------------------------------------
# Model classes (verbatim from live_demo.py Step 3)
# ----------------------------------------------------------------
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
    """Greedy CTC decode — real decode, no caching."""
    ids, out, prev = logits.argmax(-1).tolist(), [], None
    for k in ids:
        if k != prev and k != blank: out.append(vocab[k])
        prev = k
    return out


def personalize(base_state, Xs, Ys, k, mode, vocab, blank,
                epochs=60, lr=1e-4, on_epoch=None):
    """
    Fine-tune the model on k examples.
    on_epoch: optional callback(epoch_1based, total_epochs) for progress.
    """
    base = LipNet(len(vocab)).to(device)
    base.load_state_dict(base_state)
    net = PersonalizedLipNet(base, mode=mode).to(device) if mode != 'full' else base
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    Xk, Yk = Xs[:k], Ys[:k]
    net.train()
    for ep in range(epochs):
        for i in range(0, len(Xk), 2):
            bx = Xk[i:i + 2].to(device).float().div_(255.)
            ys = Yk[i:i + 2]
            tgt = torch.cat([torch.tensor(t) for t in ys]).to(device)
            tl = torch.tensor([len(t) for t in ys], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            out = net(bx)
            il = torch.full((len(ys),), out.shape[1], dtype=torch.long)
            loss = ctc(out.log_softmax(2).permute(1, 0, 2), tgt, il, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        if on_epoch is not None:
            on_epoch(ep + 1, epochs)
    net.eval()
    return net


def transcribe(clip, model, vocab, blank):
    """
    Run a REAL forward pass on the clip — no caching, no simulation.
    Returns a list of predicted words.
    """
    model.eval()
    bx = torch.from_numpy(clip).unsqueeze(0).to(device).float().div_(255.)
    with torch.no_grad():
        hyp = ctc_decode(model(bx)[0].float().cpu(), blank, vocab)
    return hyp


# ----------------------------------------------------------------
# Face detection + mouth crop  (from live_demo.py record_and_crop)
# ----------------------------------------------------------------
def crop_mouth(frames):
    """
    Detect face, crop mouth ROI, resample to 75 frames.
    Returns (np.ndarray shape (75,50,100), None) on success,
    or (None, error_message_string) on failure.
    """
    if len(frames) < 10:
        return None, "Recording too short — try again."

    # One stable face box for the whole clip
    boxes = []
    for i in np.linspace(0, len(frames) - 1, 8).astype(int):
        det = CASCADE.detectMultiScale(
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 1.1, 5)
        if len(det):
            boxes.append(det[0])
    if not boxes:
        return None, ("No face detected — check lighting and camera "
                      "angle, then try again.")

    x, y, w, h = np.median(np.array(boxes), axis=0).astype(int)
    H, W = frames[0].shape[:2]
    y1, y2 = max(0, y + int(h * 0.58)), min(H, y + int(h * 1.02))
    x1, x2 = max(0, x + int(w * 0.16)), min(W, x + int(w * 0.84))
    if y2 - y1 < 10 or x2 - x1 < 10:
        return None, "Mouth crop region too small — move closer to camera."

    roi = [cv2.resize(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                      (OUT_W, OUT_H)) for f in frames]
    roi = np.array(roi, dtype=np.uint8)

    # Resample to exactly 75 frames
    idx = np.linspace(0, len(roi) - 1, T_LEN).round().astype(int)
    return roi[idx], None


# ----------------------------------------------------------------
# Demo logging
# ----------------------------------------------------------------
def log_demo_run(entry: dict):
    """Append a timestamped record to demo_log.json."""
    entries = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []
    entries.append(entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(entries, f, indent=2)


# ================================================================
# Main application
# ================================================================
class VSRApp(tk.Tk):
    """Two-tab Tkinter UI: Baseline transcription + Personalization."""

    def __init__(self):
        super().__init__()
        self.title("VSR Live Demo — Speaker-Adaptive Visual Speech Recognition")
        self.geometry("1020x750")
        self.minsize(800, 600)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Load model checkpoint ----
        if not os.path.exists(CKPT):
            messagebox.showerror(
                "Model not found",
                f"source_model.pt not found at:\n{CKPT}\n\n"
                f"Run this script from your VSR_project folder.")
            self.destroy()
            return

        ck = torch.load(CKPT, map_location=device, weights_only=False)
        self.vocab = list(ck['vocab'])
        self.V = len(self.vocab)
        self.BLANK = self.V
        self.base_state = ck['model']

        # Build baseline model (always loaded, never mutated)
        self.base_model = LipNet(self.V).to(device)
        self.base_model.load_state_dict(self.base_state)
        self.base_model.eval()

        # Single-word personalization state
        self.personalized_model = None
        self.training_clips = []    # list of np arrays (75,50,100)
        self.training_labels = []   # list of [word_index]

        # AAC Phrase Personalization state
        self.phrase_enrollment = PhraseEnrollment()
        self.enrolled_phrases = ["I need water", "Call my mother", "I am in pain"]
        self.phrase_linear_head = None
        self.phrase_list_trained = []

        # ---- Open camera ----
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open webcam (index 0).\n\n"
                "Make sure no other application is using the camera\n"
                "and that a camera is connected.")
            self.destroy()
            return

        # Recording state — all reads happen in the main-thread after() loop
        self._running = True
        self._in_countdown = False
        self._recording = False
        self._countdown_start = 0.0
        self._record_start = 0.0
        self._recorded_frames = []
        self._record_callback = None   # called when recording finishes

        # ---- Build UI ----
        self._build_ui()

        # ---- Start camera preview loop ----
        self._update_preview()

    # ============================================================
    # UI construction
    # ============================================================
    def _create_scrollable_tab(self, tab_title):
        """Helper to create a scrollable tab inside the notebook."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tab_title)

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas)

        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return content_frame

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[14, 6],
                        font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 10))
        style.configure('Result.TLabel', font=('Consolas', 11))
        style.configure('ResultBold.TLabel',
                        font=('Consolas', 11, 'bold'))

        # ---- Notebook ----
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(padx=10, pady=(10, 4), fill='both', expand=True)

        self._build_baseline_tab()
        self._build_phrase_tab()
        self._build_personalization_tab()

        # ---- Footer ----
        dev_text = (f"Device: {device}  │  Vocab: {self.V} words  │  "
                    f"Log: demo_log.json")
        ttk.Label(self, text=dev_text, font=('Segoe UI', 8),
                  foreground='gray').pack(pady=(2, 8))

    # ---- Tab 1: Baseline ----
    def _build_baseline_tab(self):
        tab = self._create_scrollable_tab("  ▶ Baseline Transcription  ")

        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Left column: Preview & Status
        left = ttk.Frame(main_frame)
        left.grid(row=0, column=0, padx=(0, 15), sticky='n')

        self.preview_1 = tk.Label(left, bg='#1a1a1a', width=480, height=360)
        self.preview_1.pack(padx=2, pady=2)

        self.status_1 = tk.StringVar(value="Ready — press Record to begin")
        ttk.Label(left, textvariable=self.status_1,
                  style='Header.TLabel', wraplength=480).pack(pady=(6, 4))

        self.btn_baseline = ttk.Button(
            left, text="🎥  Record & Transcribe",
            command=self._baseline_record)
        self.btn_baseline.pack(pady=4)

        # Right column: Prediction display & Vocab
        right = ttk.Frame(main_frame)
        right.grid(row=0, column=1, sticky='nsew')
        main_frame.columnconfigure(1, weight=1)

        pf = ttk.LabelFrame(right, text="Prediction", padding=10)
        pf.pack(fill='x', pady=(0, 10))

        self.pred_raw_1 = tk.StringVar(value="—")
        self.pred_grammar_1 = tk.StringVar(value="—")

        ttk.Label(pf, text="Raw CTC:",
                  font=('Segoe UI', 9, 'bold')).grid(
            row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Label(pf, textvariable=self.pred_raw_1,
                  style='Result.TLabel', wraplength=380).grid(
            row=0, column=1, sticky='w', padx=6, pady=4)

        ttk.Label(pf, text="Grammar:",
                  font=('Segoe UI', 9, 'bold')).grid(
            row=1, column=0, sticky='w', padx=6, pady=4)
        lbl_gram = ttk.Label(pf, textvariable=self.pred_grammar_1,
                             style='ResultBold.TLabel', wraplength=380)
        lbl_gram.grid(row=1, column=1, sticky='w', padx=6, pady=4)
        lbl_gram.configure(foreground='#006600')

        pf.columnconfigure(1, weight=1)

        vf = ttk.LabelFrame(right, text="Vocabulary Reference", padding=10)
        vf.pack(fill='both', expand=True)

        vocab_str = "\n".join([", ".join(self.vocab[i:i+6]) for i in range(0, len(self.vocab), 6)])
        ttk.Label(vf, text=vocab_str, font=('Consolas', 9),
                  foreground='#444444').pack(anchor='w')

    # ---- Tab 2: AAC Phrase Classification ----
    def _build_phrase_tab(self):
        tab = self._create_scrollable_tab("  💬 Personalized Phrases (AAC)  ")

        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Left column: Preview & Status
        left = ttk.Frame(main_frame)
        left.grid(row=0, column=0, padx=(0, 15), sticky='n')

        self.preview_2 = tk.Label(left, bg='#1a1a1a', width=480, height=360)
        self.preview_2.pack(padx=2, pady=2)

        self.status_aac = tk.StringVar(
            value="Enroll custom phrases, record examples, then build recognizer.")
        ttk.Label(left, textvariable=self.status_aac,
                  style='Header.TLabel', wraplength=480).pack(pady=(6, 4))

        # Right column: Enrollment, Build & Recognition
        right = ttk.Frame(main_frame)
        right.grid(row=0, column=1, sticky='nsew')
        main_frame.columnconfigure(1, weight=1)

        # 1. Enroll Phrases
        ef = ttk.LabelFrame(right, text="1. Enroll Custom AAC Phrases", padding=8)
        ef.pack(fill='x', pady=(0, 8))

        entry_frame = ttk.Frame(ef)
        entry_frame.pack(fill='x', pady=(0, 6))

        ttk.Label(entry_frame, text="Phrase:").pack(side='left', padx=(0, 4))
        self.phrase_entry = ttk.Entry(entry_frame, width=24)
        self.phrase_entry.pack(side='left', padx=(0, 6))
        self.btn_add_phrase = ttk.Button(entry_frame, text="➕ Add Phrase", command=self._add_phrase)
        self.btn_add_phrase.pack(side='left')

        list_frame = ttk.Frame(ef)
        list_frame.pack(fill='x', pady=2)

        self.phrase_listbox = tk.Listbox(list_frame, height=4, selectmode='single', font=('Segoe UI', 9))
        self.phrase_listbox.pack(side='left', fill='both', expand=True)

        lb_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.phrase_listbox.yview)
        lb_scroll.pack(side='right', fill='y')
        self.phrase_listbox.config(yscrollcommand=lb_scroll.set)

        self.btn_remove_phrase = ttk.Button(ef, text="🗑 Remove Selected", command=self._remove_phrase)
        self.btn_remove_phrase.pack(anchor='w', pady=(4, 0))

        # Populate initial listbox items
        self._update_phrase_listbox()

        # 2. Record Examples
        rf = ttk.LabelFrame(right, text="2. Record Training Examples", padding=8)
        rf.pack(fill='x', pady=(0, 8))

        self.btn_record_phrase_example = ttk.Button(
            rf, text="📹  Record Example for Selected Phrase",
            command=self._record_phrase_example)
        self.btn_record_phrase_example.pack(anchor='w', padx=4, pady=2)

        ttk.Label(rf, text="Record 3-5 examples per phrase for optimal accuracy.",
                  font=('Segoe UI', 8), foreground='gray').pack(anchor='w', padx=4)

        # 3. Build Recognizer
        bf = ttk.LabelFrame(right, text="3. Build Recognizer", padding=8)
        bf.pack(fill='x', pady=(0, 8))

        self.btn_build_phrase_recognizer = ttk.Button(
            bf, text="⚡  Build Recognizer (Layer 1 + Layer 2)",
            command=self._build_phrase_recognizer)
        self.btn_build_phrase_recognizer.pack(anchor='w', padx=4, pady=2)

        self.phrase_progress_var = tk.DoubleVar(value=0)
        self.phrase_progress_bar = ttk.Progressbar(
            bf, variable=self.phrase_progress_var, maximum=100, length=380)
        self.phrase_progress_bar.pack(anchor='w', padx=4, pady=2)

        # 4. Phrase Recognition
        cf = ttk.LabelFrame(right, text="4. Phrase Recognition", padding=8)
        cf.pack(fill='x', pady=(0, 4))

        self.btn_recognize_phrase = ttk.Button(
            cf, text="🎤  Record & Recognize Phrase",
            command=self._recognize_phrase)
        self.btn_recognize_phrase.pack(anchor='w', padx=4, pady=(2, 6))

        res_frame = ttk.LabelFrame(cf, text="Ranked Matches (Top-3)", padding=6)
        res_frame.pack(fill='x', padx=4, pady=2)

        self.phrase_result_1 = tk.StringVar(value="1. —")
        self.phrase_result_2 = tk.StringVar(value="2. —")
        self.phrase_result_3 = tk.StringVar(value="3. —")

        lbl1 = ttk.Label(res_frame, textvariable=self.phrase_result_1, style='ResultBold.TLabel', wraplength=360)
        lbl1.pack(anchor='w', pady=1)
        lbl1.configure(foreground='#006600')

        lbl2 = ttk.Label(res_frame, textvariable=self.phrase_result_2, style='Result.TLabel', wraplength=360)
        lbl2.pack(anchor='w', pady=1)
        lbl2.configure(foreground='#333333')

        lbl3 = ttk.Label(res_frame, textvariable=self.phrase_result_3, style='Result.TLabel', wraplength=360)
        lbl3.pack(anchor='w', pady=1)
        lbl3.configure(foreground='#666666')

    # ---- Tab 3: Single-Word Adaptation ----
    def _build_personalization_tab(self):
        tab = self._create_scrollable_tab("  🔧 Single-Word Adaptation  ")

        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill='both', expand=True)

        # Left column: Preview & Status
        left = ttk.Frame(main_frame)
        left.grid(row=0, column=0, padx=(0, 15), sticky='n')

        self.preview_3 = tk.Label(left, bg='#1a1a1a', width=480, height=360)
        self.preview_3.pack(padx=2, pady=2)

        self.status_2 = tk.StringVar(
            value="Collect training examples, then personalize")
        ttk.Label(left, textvariable=self.status_2,
                  style='Header.TLabel', wraplength=480).pack(pady=(6, 4))

        # Right column: Controls & Comparison
        right = ttk.Frame(main_frame)
        right.grid(row=0, column=1, sticky='nsew')
        main_frame.columnconfigure(1, weight=1)

        # 1. Training controls
        tf = ttk.LabelFrame(right, text="1. Collect Training Examples", padding=8)
        tf.pack(fill='x', pady=(0, 8))

        ttk.Label(tf, text="Word to say:").grid(
            row=0, column=0, padx=6, pady=4, sticky='w')
        self.label_var = tk.StringVar(value=self.vocab[0])
        self.label_combo = ttk.Combobox(
            tf, textvariable=self.label_var,
            values=self.vocab, state='readonly', width=12)
        self.label_combo.grid(row=0, column=1, padx=6, pady=4)

        self.collected_var = tk.StringVar(value="Collected: 0 examples")
        ttk.Label(tf, textvariable=self.collected_var,
                  font=('Segoe UI', 9, 'bold')).grid(
            row=0, column=2, padx=10, pady=4)

        btn_box = ttk.Frame(tf)
        btn_box.grid(row=1, column=0, columnspan=3, pady=4, sticky='w')

        self.btn_record_train = ttk.Button(
            btn_box, text="📹  Record Training Example",
            command=self._personalize_record_example)
        self.btn_record_train.pack(side='left', padx=(6, 8))

        self.btn_clear = ttk.Button(
            btn_box, text="🗑  Clear All",
            command=self._clear_training_data)
        self.btn_clear.pack(side='left')

        # 2. Fine-tune controls
        ff = ttk.LabelFrame(right, text="2. Fine-Tune Model", padding=8)
        ff.pack(fill='x', pady=(0, 8))

        self.btn_personalize = ttk.Button(
            ff, text="🔧  Personalize (Fine-Tune on Your Examples)",
            command=self._run_personalization)
        self.btn_personalize.pack(anchor='w', padx=6, pady=4)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            ff, variable=self.progress_var, maximum=100, length=380)
        self.progress_bar.pack(anchor='w', padx=6, pady=2)

        # 3. Compare controls
        cf = ttk.LabelFrame(right, text="3. Compare Performance (Before vs After)", padding=8)
        cf.pack(fill='x', pady=(0, 8))

        self.btn_compare = ttk.Button(
            cf, text="🎤  Record & Compare (Before vs After)",
            command=self._personalize_compare)
        self.btn_compare.pack(anchor='w', padx=6, pady=4)

        self.pred_before_raw = tk.StringVar(value="—")
        self.pred_before_gram = tk.StringVar(value="—")
        self.pred_after_raw = tk.StringVar(value="—")
        self.pred_after_gram = tk.StringVar(value="—")

        grid_frame = ttk.Frame(cf)
        grid_frame.pack(fill='x', padx=6, pady=2)

        row = 0
        for label_text, var, color, bold in [
            ("BEFORE (raw):",     self.pred_before_raw,  '#993300', False),
            ("BEFORE (grammar):", self.pred_before_gram, '#993300', False),
            ("AFTER (raw):",      self.pred_after_raw,   '#006600', False),
            ("AFTER (grammar):",  self.pred_after_gram,  '#006600', True),
        ]:
            ttk.Label(grid_frame, text=label_text,
                      font=('Segoe UI', 9, 'bold')).grid(
                row=row, column=0, sticky='w', padx=(0, 6), pady=2)
            sty = 'ResultBold.TLabel' if bold else 'Result.TLabel'
            lbl = ttk.Label(grid_frame, textvariable=var, style=sty,
                            wraplength=340)
            lbl.grid(row=row, column=1, sticky='w', padx=4, pady=2)
            lbl.configure(foreground=color)
            row += 1

        grid_frame.columnconfigure(1, weight=1)

        # 4. Save model
        sf = ttk.LabelFrame(right, text="4. Save Checkpoint", padding=8)
        sf.pack(fill='x', pady=(0, 4))

        self.btn_save = ttk.Button(
            sf, text="💾  Save Personalized Model",
            command=self._save_personalized_model, state='disabled')
        self.btn_save.pack(anchor='w', padx=6, pady=2)

    # ============================================================
    # Camera preview loop  (runs in main thread via after())
    # ============================================================
    def _update_preview(self):
        if not self._running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.after(33, self._update_preview)
            return

        # ---- Handle countdown / recording state transitions ----
        now = time.time()

        if self._in_countdown:
            elapsed = now - self._countdown_start
            if elapsed >= COUNTDOWN_SECONDS:
                # Countdown done → start recording
                self._in_countdown = False
                self._recording = True
                self._record_start = now
                self._recorded_frames = []

        if self._recording:
            self._recorded_frames.append(frame.copy())
            elapsed = now - self._record_start
            if elapsed >= RECORD_SECONDS:
                # Recording done → process in worker thread
                self._recording = False
                frames = list(self._recorded_frames)
                self._recorded_frames = []
                cb = self._record_callback
                self._record_callback = None
                threading.Thread(
                    target=self._process_recording,
                    args=(frames, cb), daemon=True).start()

        # ---- Draw overlay text ----
        disp = frame.copy()
        if self._in_countdown:
            remaining = max(0, COUNTDOWN_SECONDS - (now - self._countdown_start))
            cv2.putText(disp, f"GET READY... {remaining:.0f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        elif self._recording:
            remaining = max(0, RECORD_SECONDS - (now - self._record_start))
            cv2.putText(disp, f"RECORDING... {remaining:.1f}s",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        # ---- Convert to Tkinter image ----
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((480, 360))
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the preview label on the visible tab
        current = self.notebook.index(self.notebook.select())
        if current == 0:
            target = self.preview_1
        elif current == 1:
            target = self.preview_2
        else:
            target = getattr(self, 'preview_3', self.preview_1)
        target.configure(image=imgtk)
        target.imgtk = imgtk  # prevent GC

        self.after(33, self._update_preview)

    # ============================================================
    # Recording helpers
    # ============================================================
    def _start_recording(self, callback):
        """Begin a countdown → record cycle.  callback(clip, err) is
        called on the main thread when done."""
        self._record_callback = callback
        self._in_countdown = True
        self._countdown_start = time.time()

    def _process_recording(self, frames, callback):
        """Worker thread: crop mouth region from recorded frames."""
        clip, err = crop_mouth(frames)
        if callback:
            self.after(0, lambda: callback(clip, err))

    # ============================================================
    # Tab 1: Baseline transcription
    # ============================================================
    def _baseline_record(self):
        self.btn_baseline.configure(state='disabled')
        self.status_1.set("Get ready to speak…")
        self.pred_raw_1.set("…")
        self.pred_grammar_1.set("…")

        def on_clip(clip, err):
            if clip is None:
                self.status_1.set(f"⚠  {err}")
                self.btn_baseline.configure(state='normal')
                return

            self.status_1.set("Running inference…")

            def infer():
                # REAL forward pass — not cached or faked
                hyp = transcribe(clip, self.base_model, self.vocab,
                                 self.BLANK)
                raw_text = (' '.join(hyp) if hyp
                            else '(nothing detected)')
                gram = grammar_constrained_decode(hyp)
                gram_text = (gram["sentence"] if gram["sentence"]
                             else '(nothing detected)')

                # Log to demo_log.json
                log_demo_run({
                    "timestamp": datetime.now().isoformat(),
                    "mode": "baseline",
                    "raw_prediction": hyp,
                    "grammar_prediction": gram["corrected_words"],
                    "grammar_changes": gram["changes"],
                    "grammar_warning": gram.get("warning"),
                })

                self.after(0, lambda: self._baseline_done(
                    raw_text, gram_text))

            threading.Thread(target=infer, daemon=True).start()

        self._start_recording(on_clip)

    def _baseline_done(self, raw_text, gram_text):
        self.pred_raw_1.set(raw_text)
        self.pred_grammar_1.set(gram_text)
        self.status_1.set("Done — press Record to try again")
        self.btn_baseline.configure(state='normal')

    # ============================================================
    # Tab 2: AAC Phrase Classification Event Handlers
    # ============================================================
    def _update_phrase_listbox(self):
        self.phrase_listbox.delete(0, tk.END)
        for phrase in self.enrolled_phrases:
            count = self.phrase_enrollment.example_count(phrase)
            self.phrase_listbox.insert(tk.END, f"{phrase} ({count} examples)")

    def _disable_phrase_buttons(self):
        for btn in (self.btn_add_phrase, self.btn_remove_phrase,
                    self.btn_record_phrase_example,
                    self.btn_build_phrase_recognizer, self.btn_recognize_phrase):
            btn.configure(state='disabled')

    def _enable_phrase_buttons(self):
        for btn in (self.btn_add_phrase, self.btn_remove_phrase,
                    self.btn_record_phrase_example,
                    self.btn_build_phrase_recognizer, self.btn_recognize_phrase):
            btn.configure(state='normal')

    def _add_phrase(self):
        phrase = self.phrase_entry.get().strip()
        if not phrase:
            return
        if phrase not in self.enrolled_phrases:
            self.enrolled_phrases.append(phrase)
            self._update_phrase_listbox()
            self.phrase_entry.delete(0, tk.END)
            self.status_aac.set(f"Added phrase: '{phrase}'")
            log_demo_run({
                "timestamp": datetime.now().isoformat(),
                "mode": "phrase_enrolled",
                "phrase": phrase
            })

    def _remove_phrase(self):
        sel = self.phrase_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        phrase = self.enrolled_phrases[idx]
        self.enrolled_phrases.pop(idx)
        self.phrase_enrollment.remove_phrase(phrase)
        self._update_phrase_listbox()
        self.status_aac.set(f"Removed phrase: '{phrase}'")

    def _record_phrase_example(self):
        sel = self.phrase_listbox.curselection()
        if not sel:
            self.status_aac.set("⚠ Select a phrase from the listbox first!")
            messagebox.showwarning("Select Phrase", "Please select a phrase from the listbox before recording.")
            return
        idx = sel[0]
        phrase = self.enrolled_phrases[idx]

        self._disable_phrase_buttons()
        self.status_aac.set(f"Get ready to speak: '{phrase}'...")

        def on_clip(clip, err):
            if clip is None:
                self.status_aac.set(f"⚠ {err}")
                self._enable_phrase_buttons()
                return

            self.status_aac.set("Extracting embedding & adding example...")

            def worker():
                self.phrase_enrollment.add_example(phrase, clip, self.base_model, device)
                count = self.phrase_enrollment.example_count(phrase)

                log_demo_run({
                    "timestamp": datetime.now().isoformat(),
                    "mode": "phrase_enrollment_example",
                    "phrase": phrase,
                    "example_count": count
                })

                self.after(0, lambda: self._on_phrase_example_added(phrase, count))

            threading.Thread(target=worker, daemon=True).start()

        self._start_recording(on_clip)

    def _on_phrase_example_added(self, phrase, count):
        self._update_phrase_listbox()
        self.status_aac.set(f"✓ Recorded example {count} for '{phrase}'")
        self._enable_phrase_buttons()

    def _build_phrase_recognizer(self):
        ready, msg = self.phrase_enrollment.is_ready(min_per_phrase=2, min_phrases=2)
        if not ready:
            self.status_aac.set(f"⚠ {msg}")
            messagebox.showwarning("Enrollment Not Ready", msg)
            return

        self._disable_phrase_buttons()
        self.phrase_progress_var.set(0)
        self.status_aac.set("Building recognizer (Layer 1 ready, training Layer 2)...")

        def on_epoch(ep, total):
            pct = (ep / total) * 100
            self.after(0, lambda p=pct, e=ep, t=total: (
                self.phrase_progress_var.set(p),
                self.status_aac.set(f"Training Layer 2 Linear Head... epoch {e}/{t}")
            ))

        def worker():
            try:
                embedding_dim = 512
                head, phrase_list = train_linear_head(
                    self.phrase_enrollment, embedding_dim=embedding_dim, epochs=30, lr=1e-3, on_epoch=on_epoch
                )
                log_demo_run({
                    "timestamp": datetime.now().isoformat(),
                    "mode": "phrase_recognizer_built",
                    "phrases": phrase_list,
                    "example_counts": {p: self.phrase_enrollment.example_count(p) for p in phrase_list}
                })
                self.after(0, lambda: self._on_phrase_recognizer_done(head, phrase_list))
            except Exception as exc:
                self.after(0, lambda: self._on_phrase_recognizer_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_phrase_recognizer_done(self, head, phrase_list):
        self.phrase_linear_head = head
        self.phrase_list_trained = phrase_list
        self.phrase_progress_var.set(100)
        self.status_aac.set("✓ Recognizer Built! (Layer 1 Prototype + Layer 2 Linear Head Ready)")
        self._enable_phrase_buttons()

    def _on_phrase_recognizer_error(self, err):
        self.status_aac.set(f"⚠ Training error: {err}")
        self._enable_phrase_buttons()

    def _recognize_phrase(self):
        if len(self.phrase_enrollment.prototypes) == 0:
            self.status_aac.set("⚠ Record at least 1 example before recognizing!")
            messagebox.showwarning("No Examples", "Please record at least 1 phrase example before recognizing.")
            return

        self._disable_phrase_buttons()
        self.status_aac.set("Get ready to lip-sign a phrase...")
        self.phrase_result_1.set("1. Recognizing...")
        self.phrase_result_2.set("2. ...")
        self.phrase_result_3.set("3. ...")

        def on_clip(clip, err):
            if clip is None:
                self.status_aac.set(f"⚠ {err}")
                self._enable_phrase_buttons()
                return

            self.status_aac.set("Classifying phrase...")

            def worker():
                proto_ranked = classify_prototype(clip, self.phrase_enrollment, self.base_model, device, top_k=3)
                linear_ranked = None
                if self.phrase_linear_head is not None:
                    linear_ranked = classify_linear(clip, self.phrase_linear_head, self.phrase_list_trained, self.base_model, device, top_k=3)

                log_demo_run({
                    "timestamp": datetime.now().isoformat(),
                    "mode": "phrase_recognition",
                    "prototype_results": proto_ranked,
                    "linear_results": linear_ranked
                })

                self.after(0, lambda: self._on_phrase_recognition_done(proto_ranked, linear_ranked))

            threading.Thread(target=worker, daemon=True).start()

        self._start_recording(on_clip)

    def _on_phrase_recognition_done(self, proto_ranked, linear_ranked):
        res_vars = [self.phrase_result_1, self.phrase_result_2, self.phrase_result_3]

        for i in range(3):
            if i < len(proto_ranked):
                p_phrase, p_score = proto_ranked[i]
                pct = max(0.0, min(100.0, p_score * 100))
                line = f"{i+1}. \"{p_phrase}\" — Prototype Sim: {pct:.1f}%"
                if linear_ranked and i < len(linear_ranked):
                    l_phrase, l_prob = linear_ranked[i]
                    line += f"  │  Linear Prob: {l_prob*100:.1f}%"
                res_vars[i].set(line)
            else:
                res_vars[i].set(f"{i+1}. —")

        top_phrase = proto_ranked[0][0] if proto_ranked else "Unknown"
        self.status_aac.set(f"✓ Recognized: \"{top_phrase}\"")
        self._enable_phrase_buttons()

    # ============================================================
    # Tab 2: Personalization — collect training examples
    # ============================================================
    def _personalize_record_example(self):
        label = self.label_var.get().strip().lower()
        if label not in self.vocab:
            self.status_2.set("⚠  Select a valid vocab word first")
            return

        self._disable_personalization_buttons()
        self.status_2.set(f"Recording training example for '{label}'…")

        def on_clip(clip, err):
            if clip is None:
                self.status_2.set(f"⚠  {err}")
                self._enable_personalization_buttons()
                return

            self.training_clips.append(clip)
            self.training_labels.append([self.vocab.index(label)])
            n = len(self.training_clips)
            self.collected_var.set(f"Collected: {n} examples")
            self.status_2.set(
                f"✓  Example {n} recorded for '{label}' — "
                f"add more or personalize")
            self._enable_personalization_buttons()

        self._start_recording(on_clip)

    def _clear_training_data(self):
        self.training_clips.clear()
        self.training_labels.clear()
        self.personalized_model = None
        self.collected_var.set("Collected: 0 examples")
        self.status_2.set("Training data cleared")
        self.btn_save.configure(state='disabled')
        self.pred_before_raw.set("—")
        self.pred_before_gram.set("—")
        self.pred_after_raw.set("—")
        self.pred_after_gram.set("—")
        self.progress_var.set(0)

    # ---- Fine-tune ----
    def _run_personalization(self):
        if len(self.training_clips) < 2:
            self.status_2.set(
                "⚠  Need at least 2 training examples to personalize")
            return

        unique_words = set(tuple(l) for l in self.training_labels)
        if len(unique_words) < 2:
            self.status_2.set(
                "⚠  Record examples of at least 2 different words — "
                "training on one repeated word will overfit badly.")
            return

        self._disable_personalization_buttons()
        self.btn_save.configure(state='disabled')
        self.progress_var.set(0)

        Xs_t = torch.from_numpy(np.stack(self.training_clips))
        Ys = list(self.training_labels)
        k = len(Xs_t)

        def on_epoch(epoch, total):
            pct = (epoch / total) * 100
            self.after(0, lambda e=epoch, t=total, p=pct:
                       self._finetune_progress(e, t, p))

        def worker():
            try:
                net = personalize(
                    self.base_state, Xs_t, Ys, k, 'adapter',
                    self.vocab, self.BLANK, epochs=15, on_epoch=on_epoch)
                self.after(0, lambda: self._finetune_done(net))
            except Exception as exc:
                self.after(0, lambda: self._finetune_error(str(exc)))

        self.status_2.set(f"Fine-tuning (adapter mode) on {k} examples…")
        threading.Thread(target=worker, daemon=True).start()

    def _finetune_progress(self, epoch, total, pct):
        self.progress_var.set(pct)
        self.status_2.set(f"Fine-tuning… epoch {epoch}/{total}")

    def _finetune_done(self, net):
        self.personalized_model = net
        self.progress_var.set(100)
        self.status_2.set(
            "✓  Personalization complete — now Record & Compare")
        self._enable_personalization_buttons()
        self.btn_save.configure(state='normal')

        log_demo_run({
            "timestamp": datetime.now().isoformat(),
            "mode": "personalization_complete",
            "training_examples": len(self.training_clips),
        })

    def _finetune_error(self, msg):
        self.status_2.set(f"⚠  Fine-tuning failed: {msg}")
        self._enable_personalization_buttons()

    # ---- Compare (before vs after) ----
    def _personalize_compare(self):
        if self.personalized_model is None:
            self.status_2.set(
                "⚠  Personalize first before comparing")
            return

        self._disable_personalization_buttons()
        self.status_2.set("Recording test clip…")
        self.pred_before_raw.set("…")
        self.pred_before_gram.set("…")
        self.pred_after_raw.set("…")
        self.pred_after_gram.set("…")

        def on_clip(clip, err):
            if clip is None:
                self.status_2.set(f"⚠  {err}")
                self._enable_personalization_buttons()
                return

            self.status_2.set("Running comparison inference…")

            def infer():
                # REAL forward passes — before AND after
                hyp_before = transcribe(
                    clip, self.base_model, self.vocab, self.BLANK)
                hyp_after = transcribe(
                    clip, self.personalized_model, self.vocab, self.BLANK)

                raw_b = (' '.join(hyp_before) if hyp_before
                         else '(nothing detected)')
                raw_a = (' '.join(hyp_after) if hyp_after
                         else '(nothing detected)')

                gram_b = grammar_constrained_decode(hyp_before)
                gram_a = grammar_constrained_decode(hyp_after)

                gram_b_text = (gram_b["sentence"] if gram_b["sentence"]
                               else '(nothing detected)')
                gram_a_text = (gram_a["sentence"] if gram_a["sentence"]
                               else '(nothing detected)')

                # Log EVERYTHING
                log_demo_run({
                    "timestamp": datetime.now().isoformat(),
                    "mode": "personalization_compare",
                    "training_examples": len(self.training_clips),
                    "before_raw": hyp_before,
                    "after_raw": hyp_after,
                    "before_grammar": gram_b["corrected_words"],
                    "after_grammar": gram_a["corrected_words"],
                    "before_grammar_changes": gram_b["changes"],
                    "after_grammar_changes": gram_a["changes"],
                })

                self.after(0, lambda: self._compare_done(
                    raw_b, raw_a, gram_b_text, gram_a_text))

            threading.Thread(target=infer, daemon=True).start()

        self._start_recording(on_clip)

    def _compare_done(self, raw_b, raw_a, gram_b, gram_a):
        self.pred_before_raw.set(raw_b)
        self.pred_before_gram.set(gram_b)
        self.pred_after_raw.set(raw_a)
        self.pred_after_gram.set(gram_a)
        self.status_2.set("Done — record again to compare another clip")
        self._enable_personalization_buttons()

    # ---- Save personalized model ----
    def _save_personalized_model(self):
        if self.personalized_model is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Personalized Model",
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialfile=(f"personalized_"
                         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"))
        if not path:
            return

        torch.save({
            'model': self.personalized_model.state_dict(),
            'vocab': self.vocab,
            'training_examples': len(self.training_clips),
            'timestamp': datetime.now().isoformat(),
        }, path)
        self.status_2.set(f"✓  Saved to {os.path.basename(path)}")

        log_demo_run({
            "timestamp": datetime.now().isoformat(),
            "mode": "save_model",
            "path": path,
            "training_examples": len(self.training_clips),
        })

    # ============================================================
    # Button state helpers
    # ============================================================
    def _disable_personalization_buttons(self):
        for btn in (self.btn_record_train, self.btn_clear,
                    self.btn_personalize, self.btn_compare):
            btn.configure(state='disabled')

    def _enable_personalization_buttons(self):
        for btn in (self.btn_record_train, self.btn_clear,
                    self.btn_personalize, self.btn_compare):
            btn.configure(state='normal')

    # ============================================================
    # Cleanup
    # ============================================================
    def _on_close(self):
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()


# ================================================================
if __name__ == '__main__':
    app = VSRApp()
    app.mainloop()
