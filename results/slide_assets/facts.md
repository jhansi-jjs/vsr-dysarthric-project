# Source Model & GRID Training Facts

> All values extracted directly from `source_model.pt`, `VSR_Complete_Project_LOCAL.ipynb`, and dataset files. No values are fabricated or guessed.

## 1. Source Model Hyperparameters & Architecture

- **Input Shape**: `(B, 75, 50, 100)` (Batch, Frames, Height, Width) grayscale `uint8` [0, 255], normalized by `.float().div_(255.)` and unsqueezed to `(B, 1, 75, 50, 100)`
- **Frontend**: 3-layer 3D-CNN (Conv3D 32 -> Conv3D 64 -> Conv3D 96), BatchNorm3D, ReLU, MaxPool3D `(1, 2, 2)`, Dropout3D(0.4)
- **Temporal Backbone**: 2-layer Bidirectional GRU (hidden size 256 per direction -> 512 bidirectional dim)
- **Classification Head**: `nn.Linear(512, 52)` (51 vocabulary tokens + 1 CTC blank token)
- **Total Parameter Count**: `12,544,980` parameters
- **Trainable Parameters**: `12,544,980` parameters (100% trainable during pretraining)
- **Dropout Rate**: `0.4` (`nn.Dropout3d(0.4)` after each 3D pooling, `nn.Dropout(0.4)` before head)
- **Batch Size**: `16` (`BS = 16`)
- **Epochs**: `80` (`EPOCHS = 80` in Cell 14; `train_hist.npy` contains exactly 80 loss values)
- **Optimizer**: `AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)`
- **Learning Rate**: Initial `3e-4`, scheduled via `CosineAnnealingLR(opt, T_max=80)`
- **Loss Function**: `nn.CTCLoss(blank=51, zero_infinity=True)` with mixed precision `GradScaler`
- **Gradient Clipping**: `clip_grad_norm_(model.parameters(), 5.0)`

## 2. Dataset & Split Specifications

- **Total Preprocessed GRID Clips**: `7859` sentence clips (75 frames x 50x100)
- **Split Strategy**: Utterance-level random permutation (`torch.randperm(seed=42)`)
- **Train Split**: `90%` = `7073` clips
- **Test Split**: `10%` = `786` clips
- **Speakers in Dataset**: 8 speakers (`s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`)
- **Total Word Occurrences in Dataset**: `47,154` words (each sentence has 6 words)
- **Vocabulary Size**: `51` words

## 3. Per-Class Word Counts in Training Corpus (GRID 51 Words)

| Word Token | Syntactic Category | Occurrences in `Y_grid` | Percentage of Corpus |
| :--- | :--- | :---: | :---: |
| `a` | Letter | 313 | 0.66% |
| `again` | Adverb | 1,963 | 4.16% |
| `at` | Preposition | 1,970 | 4.18% |
| `b` | Letter | 314 | 0.67% |
| `bin` | Command | 1,893 | 4.01% |
| `blue` | Color | 1,964 | 4.17% |
| `by` | Preposition | 1,975 | 4.19% |
| `c` | Letter | 316 | 0.67% |
| `d` | Letter | 310 | 0.66% |
| `e` | Letter | 313 | 0.66% |
| `eight` | Digit (0-9) | 786 | 1.67% |
| `f` | Letter | 313 | 0.66% |
| `five` | Digit (0-9) | 785 | 1.66% |
| `four` | Digit (0-9) | 787 | 1.67% |
| `g` | Letter | 313 | 0.66% |
| `green` | Color | 1,972 | 4.18% |
| `h` | Letter | 311 | 0.66% |
| `i` | Letter | 314 | 0.67% |
| `in` | Preposition | 1,959 | 4.15% |
| `j` | Letter | 314 | 0.67% |
| `k` | Letter | 316 | 0.67% |
| `l` | Letter | 319 | 0.68% |
| `lay` | Command | 1,970 | 4.18% |
| `m` | Letter | 317 | 0.67% |
| `n` | Letter | 315 | 0.67% |
| `nine` | Digit (0-9) | 786 | 1.67% |
| `now` | Adverb | 1,965 | 4.17% |
| `o` | Letter | 314 | 0.67% |
| `one` | Digit (0-9) | 786 | 1.67% |
| `p` | Letter | 312 | 0.66% |
| `place` | Command | 2,012 | 4.27% |
| `please` | Adverb | 1,968 | 4.17% |
| `q` | Letter | 315 | 0.67% |
| `r` | Letter | 314 | 0.67% |
| `red` | Color | 1,966 | 4.17% |
| `s` | Letter | 319 | 0.68% |
| `set` | Command | 1,984 | 4.21% |
| `seven` | Digit (0-9) | 786 | 1.67% |
| `six` | Digit (0-9) | 780 | 1.65% |
| `soon` | Adverb | 1,963 | 4.16% |
| `t` | Letter | 317 | 0.67% |
| `three` | Digit (0-9) | 792 | 1.68% |
| `two` | Digit (0-9) | 784 | 1.66% |
| `u` | Letter | 317 | 0.67% |
| `v` | Letter | 311 | 0.66% |
| `white` | Color | 1,957 | 4.15% |
| `with` | Preposition | 1,955 | 4.15% |
| `x` | Letter | 313 | 0.66% |
| `y` | Letter | 315 | 0.67% |
| `z` | Letter | 314 | 0.67% |
| `zero` | Digit (0-9) | 787 | 1.67% |
