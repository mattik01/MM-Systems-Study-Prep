# Poster Content — Project 14
## Niknazar & Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
### IEEE TPAMI, vol. 46, no. 7, pp. 5044-5061, July 2024

> **How to use this document:** Each section contains a text draft (longer than what fits on the poster — edit down to your taste), visualization recommendations with specific paper figure numbers, and course-connection notes for the MM-Systems angle. Sections are ordered as they would appear on the poster from top to bottom.

---

## Section 1: Explainability in Deep Learning

### Text Draft

Deep learning has achieved remarkable accuracy across medical signal analysis — often matching or exceeding human experts. But these models are **opaque**: a neural network can classify with high confidence while offering no account of its reasoning. In clinical practice this is not an abstract concern — it is a deployment blocker. Clinicians cannot blindly trust unexplainable decisions, and regulations increasingly mandate that AI in clinical pathways be accountable (EU AI Act 2024/1689, Art. 13 & Annex III; FDA, 2025, App. B).

**Three related but distinct concepts define the space** (Ali et al., 2023):
- **Transparency** — a white-box model property: the internal logic of the model can be inspected and understood directly. A linear model is transparent; a standard deep CNN is not.
- **Interpretability** — for developers and technical users: understanding *how* and *where* the model gets its results, enabling debugging and trust in the model's mechanism.
- **Explainability** — for end-users (clinicians, patients, regulators): communicating *why* the model made a specific decision in a way that builds trust and supports action.

Most current DL systems are black-box: high accuracy, low transparency, requiring post-hoc tools (Grad-CAM, SHAP, LIME) to approximate explainability after the fact. But post-hoc methods have a fundamental limit in clinical settings — a heatmap saying "this signal region mattered" does not tell a sleep technician *which waveform* was detected. The explanation fails to speak clinical vocabulary, and therefore fails to support clinical judgment.

What is needed is a **gray-box** approach (Ali et al., Fig. 1): a model whose critical components are transparent by design, enabling genuine interpretability for developers and genuine explainability for clinicians — not as a post-hoc gloss, but as a structural property. This paper delivers exactly this via **Gabor kernels** as the constrained first layer: transparent (parameters are directly inspectable), interpretable (each kernel maps to a named clinical waveform such as spindle or slow wave), and explainable (individual epoch decisions can be traced to specific waveform detections expressed in standard AASM vocabulary). Critically, this does not sacrifice accuracy — constrained, interpretable representations **outperform** unconstrained ones (+9% over standard CNN).

### Visualizations
- **White/gray/black-box spectrum** (adapted from Ali et al. Fig. 1): show where prior DL sleep models sit (black-box) vs. this paper (gray-box with white-box first layer). Clean and conceptually grounding.
- Or: a three-column mini-table — Transparency / Interpretability / Explainability × "Prior DL" vs. "This paper."

### Course Connection
- **Slides 10-14 (Post-WIMP medical applications):** Sleep monitoring is a clinical sensing application, analogous to rehabilitation and medical sensor examples in the course.

### Poster Hint
Your hook. The three-term distinction takes only 3 bullet points and immediately shows conceptual precision. Key message: black-box DL fails clinical deployment — this paper builds a gray-box model where the key first layer is white-box, achieving interpretability and explainability structurally. Keep to 4-6 sentences on the poster itself.

### References
- Ali et al. (2023), "Explainable Artificial Intelligence (XAI): What we know and what is left to attain Trustworthy Artificial Intelligence." *Information Fusion*, 99, 101805.

---

## Section 2: The Sleep Scoring Problem

### Text Draft

#### The Five Stages of Sleep

Sleep is not a uniform state but a structured cycle of distinct physiological stages, each identifiable by characteristic signal patterns across brain activity (EEG), eye movements (EOG), and muscle tone (EMG). The gold standard for measurement is **polysomnography (PSG)**. The American Academy of Sleep Medicine (AASM) defines five stages, each requiring its own multimodal evidence:

| Stage | Key EEG Signature | EOG | EMG | Meaning |
|-------|-------------------|-----|-----|---------|
| **Wake** | Alpha (8–13 Hz) | Blinks (0.5–2 Hz); reading or rapid eye movements | Variable; higher than sleep stages | Alert / drowsy |
| **N1** | Low-amplitude mixed, predominantly 4–7 Hz | Slow rolling eye movements | Slightly reduced | Light sleep onset |
| **N2** | Sleep spindles (11–16 Hz, typically 12–14 Hz); K-complexes | Typically absent | Reduced | Stable light sleep (~50% of night) |
| **N3 (SWS)** | Slow waves, 0.5–2 Hz, >75 µV (≥20% of epoch) | Absent | Low | Deep / restorative |
| **REM** | Low-amplitude mixed (similar to N1); sawtooth waves (2–6 Hz) | Rapid conjugate eye movements | Lowest chin EMG tone (atonia) | Dreaming / memory consolidation |

A typical night cycles through N1-N2-N3-N2-REM four to six times, each cycle ~90 minutes, with REM lengthening toward morning.

#### Expert Scoring: Necessary but Laborious

Trained technicians score these recordings epoch by epoch in **30-second windows** following AASM waveform criteria: spindle present → N2; slow waves >20% of epoch → N3; rapid eye movements + atonia → REM. This requires domain expertise — but even experts do not fully agree. Overall inter-rater agreement is ~80%; for N1, where EEG resembles both Wake and REM, it drops to 52-60%. A single overnight recording produces ~1,000 epochs; a busy sleep lab generates tens of thousands per night. The scale and subjectivity create strong motivation for automation.

#### DL Outperforms — But Inherits the Black-Box Problem

Automated sleep staging has progressed from classical ML (SVMs, HMMs, hand-crafted spectral features — interpretable but limited) to deep learning (CNN+BiLSTM directly on raw EEG). DL models have consistently outperformed classical approaches: DeepSleepNet (2017) achieved 82% accuracy; by 2022, models like XSleepNet reached 84-88%. Yet this is exactly where Section 1's problem re-enters — the better DL gets at staging, the more opaque its reasoning becomes. Sleep technicians are trained to find spindles, slow waves, and eye movements; a model that cannot express its decisions in those terms cannot be integrated into clinical practice, no matter how accurate its aggregate numbers are.

### Visualizations
- **Stage table above** (poster-worthy — shows multimodality requirement directly: N1 vs REM cannot be separated by EEG alone)
- **Hypnogram:** Stage cycling across a night (time on x-axis, stages on y-axis)
- **Comparison table (condensed from paper Table 6):**

| Method | Year | ACC (%) | kappa |
|--------|------|---------|-------|
| DeepSleepNet | 2017 | 82.0 | 0.76 |
| XSleepNet | 2020 | 84.6 | 0.78 |
| **Proposed** | **2024** | **93.94** | **0.88** |

### Course Connection
- **Unitizing / Interval Coding (Slides 137-145):** 30-second epochs = textbook interval coding, same stationarity assumption as MFCC speech frames.
- **Feature design from domain knowledge (Lecture 3 — Laban Effort):** AASM rules operationalize clinical expertise into computable signal features — same philosophy as Laban movement theory → kinematic feature groups.
- **CARE / Complementarity:** The stage table makes this immediately visible — no single modality suffices; N1 vs REM requires EOG+EMG to disambiguate EEG.
- **ML classification methods (Lecture 3):** Positions the paper in the classical ML → DL → interpretable DL taxonomy.

### Poster Hint
The stage table is the anchor visual — it establishes the multimodal nature of the problem at a glance. End the section by explicitly looping back to Section 1: DL solves the scale/accuracy problem but imports the explainability problem. The paper's contribution is resolving both simultaneously.

---

## Section 3: "Infusing Expert Knowledge" — What It Means and How It's Done

### Text Draft

**This is the core contribution of the paper. This must be crystal clear on the poster and in your presentation.**

The phrase "infusing expert knowledge" means: the network's architecture is designed so that its first processing layer can only learn features that have direct physiological meaning — specifically, waveforms that match the types of patterns sleep experts look for.

#### The Mechanism: Gabor Kernels

The key architectural choice is using **Gabor functions** as the network's first convolutional layer, instead of standard unconstrained convolutional filters.

A standard 1D convolutional filter is a vector of arbitrary learned weights — it can represent any pattern, but the learned pattern has no guaranteed physical interpretation. You cannot look at a standard CNN filter and say "this detects spindles."

A **Gabor function** is a cosine wave modulated by a Gaussian envelope:

```
G(t*) = exp(-pi * (t* - u)^2 / |sigma|) * cos(2*pi*f*t*)
```

It has exactly **three learnable parameters**:

| Parameter | What it controls | Clinical meaning |
|-----------|-----------------|------------------|
| **f** (frequency) | Oscillation rate of the cosine | Maps directly to EEG bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), sigma/spindle (12-16 Hz), beta (>16 Hz) |
| **sigma** (bandwidth) | Width of the Gaussian envelope | Controls duration: narrow = short burst like a spindle (0.5-2s); wide = sustained oscillation like a slow wave |
| **u** (time offset) | Center position in the 2-second window | Where in time the filter is most sensitive |

#### Why This Is "Expert Knowledge Infusion"

Because the Gabor function is constrained to be a localized oscillation, every kernel the network learns is guaranteed to be a **frequency-band filter with a specific time scale**. After training, you can inspect each kernel and directly identify what it detects:

| Learned Kernel | Frequency | Bandwidth | Matches | Clinical Meaning |
|---------------|-----------|-----------|---------|------------------|
| Kernels 3, 18 | ~1 Hz | Wide | Slow oscillations | Deep sleep (N3/SWS) marker |
| Kernels 4, 23 | ~4 Hz | Medium | Theta waves | REM-associated activity |
| Kernel 24 | ~15 Hz | Narrow | Sleep spindles | N2 marker (0.5-2s bursts) |
| Various | 8-13 Hz | Medium | Alpha rhythm | Wakefulness indicator |

This is fundamentally different from post-hoc interpretability (e.g., Grad-CAM applied to a black-box CNN):
- **Grad-CAM** tells you *where* the network looked (which pixels/time-points).
- **Gabor kernels** tell you *what waveform* the network detected — which is clinically meaningful.

The interpretability is **built into the architecture**, not an afterthought. The network physically cannot learn uninterpretable features in its first layer.

#### Kernel Configuration

- **32 Gabor kernels** for EEG (covers the full range of clinically relevant frequency bands)
- **8 Gabor kernels** for EOG (eye movements occupy a narrower frequency range)
- Time window: +/-1 second (2 seconds total at 100 Hz = 200 samples per kernel)
- All parameters (f, sigma, u) are **learned during training** via backpropagation — initialized, then optimized

#### The "Infusion" in Summary

Expert knowledge is not injected as explicit rules or auxiliary loss terms. It is encoded as an **architectural inductive bias**: by constraining the first layer to Gabor functions, the network's learned representations are forced into a space where every feature maps to a clinical waveform category. The network learns *which* waveforms matter and *how much* — but it cannot learn features that are clinically uninterpretable.

### Visualizations
- **Paper Figure 1 (ESSENTIAL — place prominently):** Shows four Gabor waveform examples:
  - (a) Gabor waveform matching spindle properties (high f, narrow sigma)
  - (b) Gabor waveform matching slow oscillation properties (low f, wide sigma)
  - (c) Gabor waveform matching K-complex properties
  - (d) Four Gabor waveforms matching main EEG frequency sub-bands (delta, theta, alpha, sigma)

- **Paper Figure 5 / Supplementary Figure S.1:** All 32 optimized EEG Gabor kernels after training (time + frequency domain). Proves the network *actually learned* clinically meaningful waveforms. Side-by-side with textbook EEG band definitions is very compelling.

- **Custom diagram suggestion:** Side-by-side comparison:
  - Left: Standard CNN filter = arbitrary learned weights = "What does this detect? Unknown."
  - Right: Gabor filter = clean oscillation in Gaussian envelope = "f=15 Hz, sigma=narrow -> spindle detector."

### Course Connection
- **Hand-crafted vs. learned features (Slides 25-26, Representation step):** The speech script says: *"We extract features — either hand-crafted features based on domain knowledge, or learned features via neural networks."* Gabor kernels are a **hybrid**: the functional form is hand-crafted (domain knowledge says sleep features are localized oscillations), but the specific parameters are learned. Best of both worlds.
- **Four-layer hierarchy (Lecture 3, Slides 53-65):** Gabor kernels sit at the boundary between Physical Signals and Low-Level Features — they convert raw time-series into frequency-band activations, which is the "feature extraction" step producing low-level features from physical signals.
- **Laban Effort features (Lecture 3):** Same philosophy — convert theoretical domain knowledge (Laban's movement theory / AASM sleep rules) into computable signal features (Effort parameters / Gabor filter activations). *(Identified in project14-plan.md Step 3: "Analogous feature design philosophy: expert domain knowledge encoded as computable signal features.")*

### Poster Hint
This should be one of the longest sections on the poster. The examiner will ask: "How exactly is expert knowledge infused?" If the poster clearly shows Gabor function -> constrained first layer -> every kernel maps to a clinical waveform, the answer is self-evident. The kernel table is poster-worthy and directly answers "what did the network learn?" in clinical terms. Emphasize the contrast with post-hoc interpretability.

---

## Section 4: Architecture — Full Explanation

### Text Draft

The system has two levels, each corresponding to a different scale of clinical reasoning:

#### Level 1: Single-Epoch Network (within one 30-second window)

This mimics how an expert examines one epoch at a time.

**Input:** One 30-second PSG epoch (EEG + EOG channels at 100 Hz).

**Layer-by-layer walkthrough (from paper Table 1):**

1. **Z-score normalization** — Standardize input amplitude across recordings
2. **Gabor Layer (EEG):** 32 trainable Gabor kernels -> 32 output channels detecting specific frequency-band activations across the 30-second window
3. **Gabor Layer (EOG):** 8 trainable Gabor kernels -> 8 output channels for eye movement patterns
4. **ReLU activation**
5. **Mixing Layer (1x1 convolution):** Combines the 40 Gabor channel outputs (32 EEG + 8 EOG), allowing the network to learn cross-channel interactions (e.g., "spindle present AND eye movements absent -> likely N2")
6. **4x repeated CNN blocks:** Each = 1D Convolution (kernel size 3) -> ReLU -> MaxPool (size 3, stride 3) -> Batch Normalization. These progressively extract higher-order features and compress the temporal dimension.
7. **Dropout (p=0.5)** — Regularization
8. **Fully Connected layers:** 256 -> 128 -> 5 (five sleep stages: Wake, S1, S2, SWS, REM)

**Output O_n:** A 5-dimensional vector (one score per stage) for epoch n.

#### Level 2: Multi-Epoch Network (across consecutive epochs)

This mimics how an expert considers surrounding epochs before finalizing a score. A single epoch viewed in isolation might look like N1, but if surrounding epochs are all REM, the expert would likely score it REM.

**Architecture:**
- **Input:** Single-epoch outputs for 9 consecutive epochs: O_{n-4}, ..., O_{n-1}, **O_n**, O_{n+1}, ..., O_{n+4} (a context window of 4.5 minutes)
- **Two Bi-LSTM layers** (hidden state size = 10) process the sequence in both directions
- The Bi-LSTM learns **transition patterns**: which stage sequences are physiologically plausible (N2->N3->N2 is normal; Wake->N3 directly is not)
- **Fully Connected layer** -> final 5-class prediction for the center epoch

#### Why Two Levels?

| Level | Mimics | Handles |
|-------|--------|---------|
| **Single-epoch** (CNN) | Expert examining one 30-second window | Within-epoch waveform detection (spindles, slow waves, eye movements) |
| **Multi-epoch** (Bi-LSTM) | Expert reviewing the surrounding context | Between-epoch transition rules (stage sequencing, eliminating physiologically implausible single-epoch errors) |

Together they mirror the two scales at which human experts operate.

#### Training Details
- **Loss function:** Standard cross-entropy (no auxiliary losses)
- **Optimizer:** Adam, learning rate = 0.000625
- **Mini-batch sampling:** Probabilistic sampling to balance stage representation (N1 and SWS are rare vs. Wake and N2)
- **Learning rate schedule:** Decreased every 5,000 iterations
- **Validation:** Every 1,000 iterations to monitor overfitting

### Visualizations
- **Paper Figure 2 (CENTRAL POSTER ELEMENT):** The full architecture diagram showing both single-epoch and multi-epoch networks. Make this the largest figure on the poster, placed centrally, with annotation arrows pointing to surrounding explanation panels.
- **Paper Table 1:** Layer-by-layer specification. Consider a simplified version next to Figure 2.
- **Paper Figure 4:** Hypnogram comparison — expert vs. single-epoch vs. multi-epoch. Powerfully demonstrates why Level 2 matters: single-epoch produces noisy stage assignments with sudden jumps; multi-epoch smooths into physiologically plausible sequences.
- **Custom annotation suggestion:** On Figure 2, add colored boxes around:
  1. Gabor layers -> label "Expert Knowledge Infusion"
  2. CNN blocks -> label "Feature Hierarchy"
  3. Bi-LSTM -> label "Transition Rules"

### Course Connection
- **Multimodal processing pipeline (Slides 15-20) — this IS the pipeline:**
  ```
  INPUT:          EEG electrodes + EOG electrodes
       |
  RAW DATA:       30-second epochs (physical signals)
       |
  REPRESENTATION: Gabor layers -> CNN features (low-level -> mid-level)
       |
  PREDICTION:     FC layers + Bi-LSTM (5-class classifier)
       |
  OUTPUT:         Sleep stage label + interpretability visualization
  ```

- **Four-layer hierarchy (Lecture 3, Slides 53-65):** The architecture directly instantiates the course hierarchy:
  - Physical signals -> raw EEG/EOG samples
  - Low-level features -> Gabor kernel activations (frequency-band power)
  - Mid-level features -> CNN-extracted higher-order patterns (spindle detection, K-complex recognition)
  - Concepts/structures -> sleep stage labels
  *(Identified in project14-plan.md Step 3.)*

- **CNN-LSTM hybrid (Slides 2034-2061):** The speech script discusses Wang et al. (2020) applying CNN-LSTM to expressive movement analysis: *"Three CNN branches for spatial feature extraction, LSTM layers for temporal modeling... demonstrating the importance of capturing both spatial patterns and temporal dynamics."* Identical paradigm: CNN for within-epoch patterns, LSTM for across-epoch dynamics.

- **Early fusion (Slides 6-8):** The Gabor layers for EEG and EOG feed into a shared mixing layer — this is **early fusion** (feature-level integration before classification). The speech script defines: *"Early fusion at the feature level."*

### Poster Hint
The architecture figure should be the visual centerpiece. Surround it with explanation panels pointing to specific components. Emphasize that the two-level design corresponds to two scales of expert reasoning — this is intuitive and memorable. The mixing layer (1x1 conv) is where multimodal fusion happens — highlight this for the MM-Systems examiner.

---

## Section 5: Data

### Text Draft

The system was evaluated on three publicly available PSG datasets:

| Dataset | Subjects | Channels | Sampling | Scoring Standard |
|---------|----------|----------|----------|------------------|
| **Sleep-EDF Expanded** | 78 (153 recordings) | Fpz-Cz EEG, Pz-Oz EEG, 1 horizontal EOG | 100 Hz | R&K + AASM modified |
| **Sleep-EDF 20** | 20 | Same as above | 100 Hz | Same |
| **DREAMS** | 20 | 3 EEG + 2 EOG + 1 EMG | 200 Hz (resampled to 100) | Both R&K and AASM |

**Three evaluation strategies:**
1. **Night-holdout 5-fold:** Train on night 1, test on night 2 of same subjects. Tests generalization to new nights from known patients.
2. **Subject-holdout 5-fold:** ~80% subjects train, ~20% test. Tests generalization to entirely new patients.
3. **Record-holdout:** 90% train, 10% test (fixed). Used for interpretability experiments.

**Class imbalance:** Wake epochs vastly outnumber N1 and SWS. Addressed via probabilistic mini-batch sampling.

### Visualizations
- The table above in compact form.
- Optional: Bar chart of epoch counts per stage (from paper Table 2) to visualize class imbalance.

### Course Connection
- **Sensory modalities (Slides 21-24):** EEG + EOG (+EMG in DREAMS) = distinct physiological modalities capturing different aspects of the same state. Directly satisfies the course definition of a multimodal system.
- **CARE — Complementarity (Lecture 2):** The speech script: *"Modalities are complementary if all must be used together to reach the goal — no single modality suffices alone."* The modality ablation confirms: EEG+EOG consistently outperforms EEG alone (+1.3% accuracy, +0.03 kappa).
- **Robustness through multimodality (Slides 3648-3654):** Three benefits: *"robustness, complementary information, and disambiguation."* The EEG-vs-EOG analysis (Figure 8) demonstrates disambiguation: S1 and REM depend significantly more on EOG than S2/SWS.

### Poster Hint
Keep short. Compact table + one sentence on evaluation strategy. The data is not the contribution.

---

## Section 6: Evaluation / Results

### Text Draft

#### Main Results (Sleep-EDF Expanded, Night-Holdout, Multi-Epoch Network)

| | Wake | S1 | S2 | SWS | REM | Overall |
|---|---|---|---|---|---|---|
| **Recall (%)** | 94.5 | 89.4 | 82.1 | 98.1 | 95.8 | - |
| **Precision (%)** | 99.8 | 49.8 | 94.4 | 80.1 | 85.2 | - |
| **F1 (%)** | 97.1 | 64.0 | 87.8 | 88.2 | 90.1 | - |
| **Accuracy** | | | | | | **92.33%** |
| **Cohen's kappa** | | | | | | **0.85** |
| **Macro F1** | | | | | | **85.41%** |

Key observations:
- **N1 remains hardest** (F1 = 64%) — reflects fundamental clinical ambiguity (inter-rater ~52-60%), not model failure. Recall of 89.4% is remarkably high.
- **SWS has highest recall (98.1%)** — slow waves are the most distinctive waveform.
- **Wake precision near-perfect (99.8%)** — almost no false wakefulness alarms.

#### Ablation Results (What Each Component Contributes)

| Configuration | Accuracy | kappa | Impact |
|---------------|----------|-------|--------|
| Full model (EEG+EOG, multi-epoch) | 93.94% | 0.88 | Baseline |
| Remove Gabor -> standard CNN | 84% | 0.77 | **-10%, -0.11 kappa** |
| Remove multi-epoch -> single-epoch only | 87.72% | 0.75 | **-6%, -0.13 kappa** |
| Remove EOG -> EEG only | 92.65% | 0.85 | -1.3%, -0.03 kappa |

**Gabor layer contributes the largest single improvement (+9% accuracy).** Multi-epoch LSTM adds +4.6%. EOG provides consistent but smaller improvement, most impactful for S1 and REM.

#### Vs. Baselines (EDF-20, LOO)

Proposed method: **93.94% accuracy, kappa = 0.88**
Best prior method (DeepSleepNet-FT): 84.6%, kappa = 0.78
**Gap: +9.3% accuracy, +0.10 kappa**

### Visualizations
- **Ablation bar chart (custom):** Three bars showing accuracy for full model vs. no Gabor vs. single-epoch only. Visually demonstrates each component's contribution.
- **Paper Table 6 (condensed):** Comparison table vs. baselines, 5-6 rows. The "proof."
- **Paper Table 5 confusion matrices:** Single-epoch vs. multi-epoch showing how LSTM corrects misclassifications (especially S1<->Wake and REM<->Wake).

### Course Connection
- **ML metrics (Lecture 3):** Accuracy, F1, and Cohen's kappa are discussed as evaluation metrics. Kappa is especially important because it accounts for chance agreement — critical with imbalanced classes.

### Poster Hint
The **ablation table** is more informative than raw results for the poster — it shows *why each component matters*. The +9% from Gabor kernels is the headline: expert knowledge infusion isn't just interpretable, it's also more accurate than unconstrained learning.

---

## Section 7: Interpretability — How the Goal Was Achieved

### Text Draft

**This is the payoff section. It answers: "Did the expert knowledge infusion actually produce interpretable outputs?"**

The paper introduces a **four-level interpretability framework**, progressively zooming from global model behavior to individual time-points:

#### Level 1: What Did the Gabor Kernels Learn? (Staging Process)

After training, the 32 EEG Gabor kernels can be visualized in both time and frequency domains. The paper shows (Figure 5 / Supplementary Figure S.1) that optimized kernels cluster around clinically known frequency bands:

- Several kernels converged to **~1 Hz** -> slow wave / delta detectors
- Several converged to **~4 Hz** -> theta detectors
- Kernel 24 converged to **~15 Hz with narrow bandwidth** -> spindle detector
- Other kernels span alpha (8-13 Hz) and beta (>13 Hz) ranges

**First interpretability payoff:** A clinician can inspect the learned filter bank and confirm it matches known EEG physiology. The network rediscovered the waveforms experts already know matter.

#### Level 2: Which Kernels Matter for Which Stages? (Stage-Specific Analysis)

Using the **Effective Functional Effect (EFF)** metric — which measures each kernel's contribution to each stage's classification — the paper shows (Figure 6):

- **Slow-wave kernels (3, 18):** Dominant impact on SWS, minimal on Wake/REM
- **Spindle kernel (24):** Highest impact on S2 and SWS — matches AASM criteria
- **Theta kernels (4, 23):** Highest impact on REM — consistent with REM theta activity
- **Alpha kernels:** Highest impact on Wake — consistent with posterior alpha rhythm

The EFF metric formula:

```
EFF_i^X(t) = GL^i(t) x Sen(t)_{GL(t)->O[class]} x theta(Sen(t)_{GL(t)->O[class]})
```

Where:
- GL^i(t) = output of Gabor kernel i at time t
- Sen = sensitivity (gradient) of the output with respect to the kernel activation
- theta = Heaviside step function (keeps only positive contributions)

#### Level 3: EEG vs. EOG — Which Modality Drives Each Stage? (Signal Analysis)

Figure 8 shows the ratio of EEG-to-EOG kernel impact per stage:
- **S1 and REM:** Significantly higher EOG dependence (p < 0.05) — consistent with eye movements being clinically diagnostic
- **S2 and SWS:** No significant modality difference — consistent with AASM criteria (no eye movements expected)

**This validates multimodal design:** the network learned to weight modalities exactly as experts do.

#### Level 4: Temporal Activation — What Happens Within a 4 Single Epoch? (Time-Series Analysis)

Figure 9 provides the most granular level: for individual epochs, EFF(t) heatmaps across time reveal *when* during the 30-second window each kernel activates:

- **S2 epoch (Fig 9a):** Spindle kernel activates at ~20-23 seconds -> detected a spindle burst exactly where one occurs in the raw signal
- **SWS epoch (Fig 9b):** Slow-wave kernels activate broadly -> sustained delta activity
- **REM epoch (Fig 9c):** Theta kernels with intermittent eye-movement kernel bursts

**This is the ultimate interpretability demonstration:** a clinician can look at a classified epoch and see exactly *which waveform* the model detected, *when* it detected it, and *how much* it contributed. This maps directly onto the visual inspection process human scorers perform.

### Visualizations
- **Paper Figure 5 / Figure S.1 (ESSENTIAL):** All 32 trained Gabor kernels (time + frequency domain). The proof that the network learned physiologically meaningful features. Place selected kernels alongside textbook EEG band definitions.
- **Paper Figure 6c/6d (ESSENTIAL):** Per-stage EFF distribution (pie charts or bar charts) showing which kernels dominate which stages. Clearest visualization of "the model uses the right features for the right stages."
- **Paper Figure 8:** EEG-vs-EOG modality impact boxplots. Compact but powerful — directly demonstrates multimodal complementarity quantitatively.
- **Paper Figure 9 (ESSENTIAL — pick one example, suggest S2/Fig 9a):** Time-series EFF heatmap for a single epoch. The "hero visualization" — shows everything at once. The S2 example with spindle detection is probably most compelling.
- **Paper Figure 7:** Box plots of EFF per kernel per stage with statistical significance markers (p<0.05). Shows kernel-stage associations are statistically robust.

### Course Connection
- **Four-layer hierarchy as interpretability levels:** The paper's four levels map onto the course hierarchy in reverse — from concepts (which stage?) down to physical signals (which time-point?). The hierarchy is used as an analytical tool for model understanding.
- **Feature design philosophy (Lecture 3 — 10 feature groups):** project14-plan.md notes: *"AASM rules -> computable signal features is the same methodology as Laban Effort features derived from movement theory."* The interpretability section proves this worked: Gabor activations actually correspond to expert-defined categories.

### Poster Hint
- This proves the paper delivered on its promise. Four levels = natural narrative arc: zoom out (what did it learn?) -> zoom in (what happens in one epoch?).
- **Figure 9 is the single most impactful visualization.** If you can only fit one interpretability figure, use this.
- **Figure 8 (EEG vs EOG) is a guaranteed exam talking point** — directly demonstrates multimodal complementarity, a core course concept.

---

## Section 8: Conclusion, Impact, and Future Work

### Text Draft

#### What Was Achieved

The paper demonstrates that expert knowledge can be structurally embedded into a deep learning architecture through Gabor kernel constraints, producing a system that is simultaneously:
- **More accurate** than prior methods (+9% over standard CNN, +9.3% over best baseline)
- **Interpretable at four levels** — from global kernel analysis to per-time-point waveform detection
- **Clinically grounded** — interpretability expressed in AASM-standard waveform vocabulary, not generic DL attribution

The key insight: constraining the network to learn within a physiologically meaningful function space (Gabor functions) does not sacrifice performance — it **improves** it. This challenges the common assumption of an accuracy-interpretability trade-off.

#### Limitations
- Evaluated only on healthy adult populations; generalization to pediatric, elderly, or clinical populations (apnea, narcolepsy) untested
- Gabor kernels capture oscillatory patterns well but may miss non-oscillatory transients (certain artifact types)
- Only EEG + EOG used; EMG not included (though AASM requires it for REM scoring via atonia)
- Hyperparameter choices (32 EEG kernels, 8 EOG kernels, +/-4 epoch context) not extensively ablated

#### Future Directions (from paper)
- Incorporating additional modalities (e.g., ECG for cardiac-influenced sleep events)
- Examining **negative gradients** — understanding which kernels suppress certain classifications
- Extending to disorder-specific scoring rules

#### Broader Impact
Published in IEEE TPAMI (top-tier venue). The Gabor-constrained-filter approach is generalizable beyond sleep to any signal classification domain where expert-defined waveforms exist: cardiac arrhythmia detection (ECG), seizure detection (EEG), speech disorder analysis.

### Visualizations
- No complex figures. Compact summary box or bullet list.
- Optional: "Future directions" diagram showing the approach extending to ECG, EMG, respiratory signals.

### Course Connection
- **Modality selection (Lecture 2):** The missing EMG limitation connects to: *"Modalities are complementary if all must be used together."* EMG would complete the AASM-standard modality set.
- **Five multimodal ML challenges (Baltrusaitis, Ahuja & Morency 2019, Slides 3640-3690):** The paper addresses:
  - **Representation challenge** (Gabor kernels as principled multimodal representation)
  - **Fusion challenge** (mixing layer for cross-modality integration)
  - **Co-learning challenge** (shared CNN layers after mixing allow EEG and EOG to inform each other)

### Poster Hint
End on a forward-looking note. The Gabor approach is generalizable — worth one sentence. "The accuracy-interpretability trade-off is false" is a strong closing statement.

---

## Summary of Key Figures to Include on Poster

| Priority | Figure | What it shows | Best placement |
|----------|--------|---------------|----------------|
| **Must have** | Fig 2 (Architecture) | Full system diagram | Center of poster |
| **Must have** | Fig 1 (Gabor examples) | How Gabor kernels represent expert waveforms | Section 4 |
| **Must have** | Fig 9 (Time-series EFF) | Per-time-point interpretability heatmap | Section 8 |
| **Must have** | Table 6 (Baselines) | Comparison vs. prior work | Section 3 or 7 |
| **Should have** | Fig 6c/d (Per-stage EFF) | Which kernels matter for which stages | Section 8 |
| **Should have** | Fig 8 (EEG vs EOG) | Multimodal complementarity proof | Section 8 |
| **Should have** | Fig 4 (Hypnogram) | Expert vs. model comparison | Section 2 or 5 |
| **Nice to have** | Fig 5/S.1 (All kernels) | Full learned filter bank | Section 4 or 8 |
| **Nice to have** | Fig 10 (Ablation curves) | Gabor prevents overfitting | Section 7 |
| **Nice to have** | AASM stage traces | Example signals per stage | Section 2 |

---

## Summary of Course Connections

| Course Concept | Slide Reference | Paper Connection |
|---------------|-----------------|------------------|
| Multimodal processing pipeline | 15-20 | The paper IS this pipeline: sensors -> features -> classification -> labels |
| Four-layer feature hierarchy | 53-65 | Architecture maps directly: raw signals -> Gabor activations -> CNN features -> stage labels |
| Early fusion | 6-8 | Mixing layer combines EEG + EOG features before classification |
| CARE / Complementarity | 35-41 | EEG + EOG are complementary; ablation confirms neither suffices alone |
| Frame-based analysis / Unitizing | 137-145 | 30-second epochs = interval coding, same as MFCC speech frames |
| Feature design from domain knowledge | Laban Effort, Lecture 3 | AASM rules -> Gabor features, same as Laban theory -> movement features |
| CNN-LSTM hybrid | 2034-2061 | CNN for spatial patterns + LSTM for temporal dynamics |
| Hand-crafted vs. learned features | 25-26 | Gabor = hybrid: hand-crafted form, learned parameters |
| ML classification & metrics | Lecture 3 | kNN/SVM/DL taxonomy; kappa for imbalanced classes |
| Robustness & disambiguation | 3648-3654 | EOG disambiguates S1/REM from S2/SWS |
| Five MM-ML challenges | 3640-3690 | Addresses Representation, Fusion, and Co-learning challenges |
| Post-WIMP medical applications | 10-14 | Sleep staging as clinical sensing application |

---

## References

All citations used in this document, formatted for poster use.

### Main Paper
1. **Niknazar, M., & Mednick, S. C.** (2024). A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(7), 5044–5061. https://doi.org/10.1109/TPAMI.2023.3327296

### Regulatory
2. **European Parliament and Council of the European Union.** (2024). Regulation (EU) 2024/1689 on Artificial Intelligence (EU AI Act). *Official Journal of the European Union*, L 2024/1689. Article 13 (Transparency and provision of information to deployers); Annex III §5(a) (High-risk classification: AI systems in medical devices regulated under MDR/IVDR).

3. **U.S. Food and Drug Administration.** (2025, January). *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations* (Draft Guidance, Docket No. FDA-2024-D-4488). Appendix B: Transparency Design Considerations. https://www.fda.gov/media/184856/download

### XAI / Explainability
4. **Ali, S., Abuhmed, T., El-Sappagh, S., Muhammad, K., Alonso-Moral, J. M., Confalonieri, R., Guidotti, R., Del Ser, J., Díaz-Rodríguez, N., & Herrera, F.** (2023). Explainable Artificial Intelligence (XAI): What we know and what is left to attain Trustworthy Artificial Intelligence. *Information Fusion*, 99, 101805. https://doi.org/10.1016/j.inffus.2023.101805

### Sleep Staging Domain
3. **Berry, R. B., Brooks, R., Gamaldo, C., Harding, S. M., Lloyd, R. M., Quan, S. F., & Troester, M. T.** (2017). *AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications* (Version 2.4). American Academy of Sleep Medicine.

### Prior DL Sleep Staging Methods (for comparison table)
4. **Supratak, A., Dong, H., Wu, C., & Guo, Y.** (2017). DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 25(11), 1998–2008.

5. **Mousavi, S., Afghah, F., & Acharya, U. R.** (2019). SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach. *PLOS ONE*, 14(5), e0216456.

6. **Phan, H., Chén, O. Y., Tran, M. C., Koch, P., Mertins, A., & De Vos, M.** (2021). XSleepNet: Multi-view sequential model for automatic sleep staging. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(9), 5903–5915.

### Multimodal ML
7. **Baltrusaitis, T., Ahuja, C., & Morency, L.-P.** (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.

### Datasets
8. **Cassani, R., Dreyfus-Cooper, J., & Bhatt, P.** Sleep-EDF Database Expanded. PhysioNet. (Dataset used in paper evaluation.)

9. **Devuyst, S., Dutoit, T., Stenuit, P., & Kerkhofs, M.** (2011). The DREAMS Databases and Assessment Algorithm. *2011 Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, 4343–4346.
