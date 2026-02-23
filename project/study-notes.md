# Study Notes — Project 14
## Niknazar & Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
### *IEEE TPAMI*, vol. 46, no. 7, pp. 5044–5061, July 2024

> **Note**: Sections marked *(verify)* are reconstructed from model training knowledge — check exact values against the paper PDF.

---

## Reading Log
- [ ] Abstract
- [ ] Introduction (Section I)
- [ ] Related Work (Section II)
- [ ] Methodology / Architecture (Section III)
- [ ] Experiments / Datasets (Section IV)
- [ ] Results (Section V)
- [ ] Discussion / Conclusion (Section VI)
- [ ] Supplementary material

---

## 1. One-Paragraph Summary
*(Fill in after reading abstract + conclusion.)*

DOI: `10.1109/TPAMI.2024.3374072` (verify on IEEE Xplore).

---

## 2. Problem Statement

**Clinical context:**
- Manual sleep staging: expert technicians score 30-second PSG epochs into **Wake, N1, N2, N3, REM** following AASM rules
- Inter-rater agreement: ~80% overall; drops to ~52–60% for N1 (most transitional stage)
- Bottleneck: expensive, slow, and subjective

**Gap in existing DL work:**
- Models match or exceed human accuracy but are black boxes
- Clinicians cannot inspect or verify decisions → no clinical trust
- No systematic incorporation of structured AASM expert knowledge

**Paper's claimed contribution:**
- Multi-level architecture that is accurate **and** interpretable
- AASM expert knowledge infused via **trainable Gabor kernels** as the network's first layer (architectural inductive bias, not auxiliary losses)
- Four-level interpretability framework with the Effective Functional Effect (EFF) metric

---

## 3. Architecture

### 3.1 Input Modalities

| Modality | Channels (Sleep-EDF) | Clinical role |
|---|---|---|
| EEG | Fpz-Cz, Pz-Oz | Brain state: spindles (N2), K-complexes (N2), slow waves (N3), mixed-frequency (N1) |
| EOG | 1 horizontal EOG | Rapid eye movements → REM detection; slow rolling movements → N1 |

**Note:** EMG is **not used** in this paper (only EEG + EOG). The DREAMS dataset includes EMG but the paper processes only EEG and EOG channels.

**Windowing**: 30-second epochs at 100 Hz → 3,000 samples per epoch.
*Same frame-based analysis as MFCC in speech (MM-Systems Lecture 5).*

### 3.2 Two-Level Architecture

The system has two levels corresponding to two scales of expert reasoning:

**Level 1 — Single-Epoch Network** (within one 30-second window):

| Step | Layer | Details |
|---|---|---|
| 1 | Z-score normalization | Standardize amplitude across recordings |
| 2 | **Gabor Layer (EEG)** | 32 trainable Gabor kernels → 32 frequency-band activation channels |
| 3 | **Gabor Layer (EOG)** | 8 trainable Gabor kernels → 8 eye-movement pattern channels |
| 4 | ReLU | — |
| 5 | **Mixing Layer (1×1 conv)** | Combines 40 Gabor outputs; learns cross-modal interactions (early fusion) |
| 6 | 4× CNN blocks | Each: 1D Conv (k=3) → ReLU → MaxPool (k=3, s=3) → BatchNorm |
| 7 | Dropout (p=0.5) | Regularization |
| 8 | FC layers | 256 → 128 → 5 (five sleep stages) |

Output O_n: 5-dimensional vector for epoch n.

**Level 2 — Multi-Epoch Network** (across consecutive epochs):
- Input: O_{n-4}, ..., O_n, ..., O_{n+4} (9 consecutive epochs = 4.5 minutes context)
- Two **Bi-LSTM layers** (hidden size = 10)
- FC layer → final 5-class prediction for center epoch
- Learns transition patterns: which stage sequences are physiologically plausible

| Level | Mimics | Handles |
|---|---|---|
| Single-epoch (CNN) | Expert examining one window | Within-epoch waveform detection |
| Multi-epoch (Bi-LSTM) | Expert reviewing surrounding context | Between-epoch transition rules |

**MM-Systems four-layer hierarchy mapping:**

| MM-Systems Layer | Architecture Component |
|---|---|
| Physical signals | Raw EEG/EOG samples |
| Low-level features | Gabor kernel activations (frequency-band power) |
| Mid-level features | CNN-extracted higher-order patterns (spindle detection, K-complex recognition) |
| Concepts/structures | Sleep stage labels (Wake/S1/S2/SWS/REM) |

### 3.3 Expert Knowledge Infusion — The Gabor Kernel Mechanism

> **CORRECTION:** Earlier versions of these notes described auxiliary losses, attention regularization, and transition probability constraints. **The actual paper uses none of these.** The loss function is simply cross-entropy. Expert knowledge is infused purely through architectural design.

The primary mechanism is using **Gabor functions** as the network's first convolutional layer instead of unconstrained filters.

**Gabor function:**
```
G(t*) = exp(-π(t* - u)² / |σ|) · cos(2πft*)
```

Three learnable parameters:
- **f** (frequency): maps to EEG bands — delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), sigma (12–16 Hz)
- **σ** (bandwidth): narrow = short burst (spindle); wide = sustained oscillation (slow wave)
- **u** (time offset): center position in the ±1 second window

**Why this is "expert knowledge infusion":** Gabor functions are constrained to be localized oscillations. Every kernel the network learns is guaranteed to be a frequency-band filter with physiological meaning. After training, each kernel can be inspected and matched to a known sleep waveform:

| Kernel | Frequency | Matches | Clinical meaning |
|---|---|---|---|
| Kernels 3, 18 | ~1 Hz (wide σ) | Slow oscillations | SWS/N3 marker |
| Kernels 4, 23 | ~4 Hz (medium σ) | Theta waves | REM-associated |
| Kernel 24 | ~15 Hz (narrow σ) | Sleep spindles | N2 marker |
| Various | 8–13 Hz | Alpha rhythm | Wakefulness |

**Configuration:** 32 EEG kernels + 8 EOG kernels. Time window: ±1s (200 samples at 100 Hz). Parameters learned via backpropagation.

**Loss function:** Standard cross-entropy only:
```
loss(O, class) = -log(exp(O[class]) / Σ_k exp(O[k]))
```

**Training:** Adam optimizer, lr = 0.000625, decreased every 5000 iterations. Minibatch = 16 with probabilistic sampling for class balance.

---

## 4. Datasets & Evaluation Protocol

| Dataset | N subjects | Channels | Sampling | Notes |
|---|---|---|---|---|
| **Sleep-EDF Expanded** | 78 (153 recordings) | Fpz-Cz EEG, Pz-Oz EEG, 1 horizontal EOG | 100 Hz | Main dataset; healthy adults; 2 nights/subject; R&K + AASM scoring |
| **Sleep-EDF 20** | 20 | Same as above | 100 Hz | Subset for LOO cross-validation comparison with prior work |
| **DREAMS** | 20 | 3 EEG + 2 EOG + 1 EMG | 200 Hz (resampled to 100) | Independent validation; both R&K and AASM scoring |

> **CORRECTION:** Earlier notes listed SHHS (~5,800), MASS (~200), and ISRUC (~100). These datasets are **not used in this paper**.

**Evaluation strategies:**
1. **Night-holdout 5-fold:** Train on night 1, test on night 2 of same subjects
2. **Subject-holdout 5-fold:** ~80% subjects train, ~20% test (new patients)
3. **Record-holdout:** 90% train, 10% test (fixed split; used for interpretability experiments)

**Metrics**: Overall accuracy (ACC), Macro F1 (MF1), Cohen's κ, per-stage Recall/Precision/F1

---

## 5. Results

### 5.1 Main Results — Sleep-EDF Expanded, Night-Holdout (from paper Table 3)

| | Wake | S1 | S2 | SWS | REM | Overall |
|---|---|---|---|---|---|---|
| **Recall (%)** | 94.48±1.01 | 89.42±4.09 | 82.08±2.46 | 98.09±0.89 | 95.78±0.46 | — |
| **Precision (%)** | 99.80±0.13 | 49.81±4.31 | 94.35±1.32 | 80.08±3.46 | 85.24±5.88 | — |
| **F1 (%)** | 97.07±0.57 | 63.95±4.52 | 87.77±1.26 | 88.16±2.43 | 90.12±3.45 | — |
| **Accuracy** | | | | | | **92.33% ± 0.59** |
| **Cohen's κ** | | | | | | **0.85 ± 0.00** |
| **Macro F1** | | | | | | **85.41% ± 1.57** |

Subject-holdout: ACC = 90.08% ± 1.63, κ = 0.81, MF1 = 80.81%

### 5.2 Comparison vs. Baselines (from paper Table 6, EDF-20 LOO)

| Method | Channels | ACC (%) | κ |
|---|---|---|---|
| DeepSleepNet (2017) | EEG | 82.0 | 0.760 |
| SleepEEGNet (2018) | EEG | 84.3 | 0.79 |
| SeqSleepNet-FT (2019) | EEG-EOG | 84.3 | 0.776 |
| DeepSleepNet-FT (2020) | EEG-EOG | 84.6 | 0.782 |
| Scattering spectrum (2020) | EEG | 84.44 | 0.784 |
| **Proposed (EEG-EOG)** | **EEG-EOG** | **93.94** | **0.88** |
| **Proposed (EEG only)** | **EEG** | **92.65** | **0.85** |

### 5.3 Ablation Results

| Configuration | ACC | κ | Impact |
|---|---|---|---|
| Full model (EEG+EOG, multi-epoch) | 93.94% | 0.88 | Baseline |
| Remove Gabor → standard CNN | 84% | 0.77 | **−10%, −0.11 κ** |
| Remove multi-epoch → single-epoch only | 87.72% | 0.75 | **−6%, −0.13 κ** |
| Remove EOG → EEG only | 92.65% | 0.85 | −1.3%, −0.03 κ |

### 5.4 Interpretability Results

The paper introduces a **four-level interpretability framework**:

1. **Level 1 — Staging Process:** Trained Gabor kernels visualized in time/frequency domain cluster around clinically known bands (slow waves ~1 Hz, theta ~4 Hz, spindles ~15 Hz, alpha 8–13 Hz). See Figures 5, S.1.
2. **Level 2 — Stage-Specific Analysis:** EFF metric shows which kernels dominate which stages (slow-wave kernels → SWS; spindle kernel 24 → S2; theta kernels → REM). See Figure 6.
3. **Level 3 — Signal Analysis:** EEG-to-EOG impact ratio shows S1 and REM depend significantly more on EOG (p<0.05), matching AASM criteria. See Figure 8.
4. **Level 4 — Time-Series:** EFF(t) heatmaps show per-time-point kernel activation within single epochs (spindle burst at ~20–23s in S2 epoch, sustained delta in SWS). See Figure 9.

---

## 6. Discussion

### 6.1 Strengths
- Clinician-verifiable: Gabor kernel activations map directly to AASM-standard waveforms — clinicians can inspect what the network learned
- State-of-the-art accuracy across Sleep-EDF and DREAMS datasets
- Gabor kernels improve both accuracy (+9%) AND interpretability simultaneously — challenges the accuracy-interpretability trade-off assumption
- Four-level interpretability framework from global kernel analysis to per-time-point waveform detection
- N1 recall of 89.4% is remarkably high given fundamental clinical ambiguity

### 6.2 Limitations
- Evaluated only on healthy adult populations; generalization to pediatric, elderly, or clinical populations untested
- Gabor kernels capture oscillatory patterns well but may miss non-oscillatory transients (certain artifacts)
- Only EEG + EOG used; **EMG not included** (though AASM requires it for REM scoring via atonia)
- Hyperparameter choices (32 EEG kernels, 8 EOG kernels, ±4 epoch context) not extensively ablated
- N1 human inter-rater agreement (~52–60%) sets a fundamental accuracy ceiling

### 6.3 Future Work (from paper Section 7)
- Incorporating additional modalities (e.g., ECG for cardiac-influenced sleep events)
- Examining **negative gradients** — understanding which kernels suppress certain classifications (currently only positive contributions analyzed via EFF)
- Extension to disorder-specific scoring rules

---

## 7. Course Connections (MM-Systems)

### 7.1 Multimodal Processing Pipeline (Slides 15–20)
```
INPUT DEVICES:   EEG electrodes + EOG electrodes
      ↓
RAW DATA:        30-second epochs (physical signals at 100 Hz)
      ↓
REPRESENTATION:  Gabor layers → CNN features (low-level → mid-level)
      ↓
PREDICTION:      FC layers + Bi-LSTM (5-class classifier)
      ↓
OUTPUT:          Sleep stage label + interpretability visualization (EFF heatmaps)
```

### 7.2 Four-Layer Hierarchy (Lecture 3)
The paper's architecture is a direct instantiation of the course hierarchy:
- **Physical signals** → raw EEG/EOG sample arrays
- **Low-level features** → Gabor kernel activations (frequency-band power per time-point)
- **Mid-level features** → CNN-extracted higher-order patterns (spindle detection, K-complex recognition)
- **Concepts and structures** → sleep stage labels (Wake/S1/S2/SWS/REM)

### 7.3 Multimodal Complementarity (Lecture 2)
EEG and EOG carry complementary information — no single modality suffices:
- EEG alone: cannot reliably distinguish S1 from REM (both low-amplitude mixed-frequency)
- EOG alone: no brain-state information for N2/N3 discrimination
- Ablation confirms: EEG+EOG outperforms EEG alone (+1.3% ACC, +0.03 κ)
- Figure 8 shows S1 and REM depend significantly more on EOG (p<0.05) while S2/SWS do not
→ Same complementarity as "Put That There" (speech + gesture): together they enable what neither can do alone.

### 7.4 Feature Design Philosophy (Lecture 3 — 10 groups of features)
AASM rules → computable signal features is the same methodology as:
- Laban Effort features derived from movement theory
- The 10 groups of kinematic low-level features derived from biomechanics
The paper operationalises clinical expert knowledge the way the course operationalises movement science theory.

### 7.5 Frame-Based Analysis (Lecture 5 — Speech)
30-second PSG epochs with fixed sampling rate = exact analogue of speech analysis frames used for MFCC extraction. Same windowing rationale: stationarity assumption within the window.

---

## 8. Exam Preparation

**Q1: What exactly is novel compared to DeepSleepNet?**
> DeepSleepNet (CNN+BiLSTM) is purely data-driven — no domain knowledge, black box. This paper: (1) replaces the first CNN layer with trainable **Gabor kernels** that are constrained to learn physiologically meaningful frequency-band filters; (2) introduces a four-level interpretability framework with the EFF metric to quantify which waveforms the model uses for each stage; (3) achieves +9% accuracy improvement over standard CNN precisely because of the Gabor inductive bias. The interpretability is structural (built into the architecture), not post-hoc.

**Q2: How do they inject expert knowledge — what is the exact mechanism?**
> Expert knowledge is infused as an **architectural inductive bias**, not through auxiliary losses or explicit rules. The first layer uses Gabor functions — cosine waves modulated by Gaussian envelopes — with three learnable parameters (frequency f, bandwidth σ, offset u). Because Gabor functions can only represent localized oscillations, every learned kernel is guaranteed to be a frequency-band filter with direct physiological meaning. After training, kernels match known sleep waveforms: ~1 Hz (slow waves/N3), ~4 Hz (theta/REM), ~15 Hz (spindles/N2). The loss function is simply cross-entropy — no auxiliary terms.

**Q3: Which modality contributes most to classification?**
> EEG is primary for all stages. EOG adds +1.3% accuracy and +0.03 κ. Crucially, Figure 8 shows S1 and REM depend significantly more on EOG than S2/SWS (p<0.05) — consistent with eye movements being clinically diagnostic only for those stages. EMG is **not used** in this paper, though AASM rules require it for full scoring.

**Q4: How is this a multimodal system in the MM-Systems sense?**
> EEG and EOG are two distinct physiological sensory modalities capturing different aspects of the same state (brain activity vs. eye movement). The system uses **early fusion** (mixing layer after Gabor outputs) to integrate them before classification — satisfying both the W3C definition ("input in more than one modality") and the course's complementarity criteria. The ablation study confirms neither modality alone suffices for optimal performance.

**Q5: What does 'interpretable' mean here, and for whom?**
> Interpretable for sleep clinicians, at four levels: (1) trained Gabor kernels can be visualized and matched to known EEG waveforms; (2) the EFF metric shows which kernels contribute to which stages; (3) EEG-vs-EOG ratios match clinical expectations per stage; (4) time-series EFF heatmaps show exactly which waveform was detected at which time-point in each epoch. This is fundamentally different from Grad-CAM — Gabor kernels tell you *what waveform* was detected (clinically meaningful), not just *where* the network looked.

**Q6: Why is constraining only the first layer to Gabor sufficient for interpretability — don't layers 2+ still reason opaquely?**
> This is a sharp challenge. The defence rests on three arguments:
>
> 1. **Information bottleneck.** By constraining layer 1 to Gabor kernels, the network is forced to decompose the raw signal into frequency-band activations before anything else can happen. Layer 2 never sees the raw signal — only Gabor outputs. The network therefore cannot learn to respond to anything the Gabor layer does not pass through. The clinical vocabulary is not merely imposed at layer 1; it limits what the entire network can ever "notice" about the input. An unconstrained CNN could learn arbitrary time-domain quirks at layer 1, and layers 2+ would reason on top of those. That pathway is closed here.
>
> 2. **EFF bridges the opaque middle.** Layers 2–N are still opaque in terms of their internal computations. But the Effective Functional Effect (EFF) metric propagates gradient information backward from the final classification decision through all intermediate layers to the first-layer Gabor activations, producing a direct attribution: "this epoch was classified as N2 because kernel 24 (~15 Hz, narrow bandwidth) was strongly active at t = 20–23 s." You do not need to understand what the intermediate layers computed internally, because EFF asks them which layer-1 features they relied on.
>
> 3. **The architecture mirrors clinical reasoning structure.** Human experts also cannot trace every step of their reasoning. They (1) decompose the signal into frequency bands perceptually, (2) recognize waveform patterns, (3) reach a stage decision. The paper's architecture follows the same order. The intermediate layers doing opaque combination is analogous to unconscious expert integration — but the inputs to that integration and the final output are both in clinical vocabulary, which is what matters for accountability.
>
> **Honest limitation:** The paper does not claim full glass-box interpretability. It does not explain *how* the deeper layers combine Gabor activations. Whether attribution back to layer 1 alone is truly sufficient is a legitimate open question — and a defensible point of critical discussion.
