# Poster Content — Project 14
## Niknazar & Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
### IEEE TPAMI, vol. 46, no. 7, pp. 5044-5061, July 2024

> **How to use this document:** Each section contains a text draft (longer than what fits on the poster — edit down to your taste), visualization recommendations with specific paper figure numbers, and course-connection notes for the MM-Systems angle. Sections are ordered as they would appear on the poster from top to bottom.

---

## Section 1: Relevancy / Why This Matters

### Text Draft

Sleep is not a passive shutdown of the brain — it is an active, structured process critical to memory consolidation, immune function, metabolic regulation, and cardiovascular health. Large-scale epidemiological studies have linked chronic sleep disruption to Alzheimer's disease, obesity, type-2 diabetes, and depression, making sleep research one of the fastest-growing fields in clinical neuroscience.

At the intersection of this clinical urgency sits a technological tension: deep learning methods have achieved remarkable accuracy in automating sleep analysis, often matching or exceeding human experts. Yet these models remain opaque — clinicians cannot inspect why a model assigned a particular sleep stage, making it impossible to verify, correct, or trust automated decisions. In a medical context where misclassification can affect treatment plans, this "black box" problem is not merely academic — it is a barrier to clinical deployment.

This paper addresses both sides: it proposes a deep learning architecture that achieves state-of-the-art sleep staging accuracy **while** providing interpretable outputs grounded in the same clinical knowledge that human experts use. The key insight is architectural: by using Gabor kernels — waveform-shaped filters whose parameters have direct physiological meaning — as the network's first layer, every learned feature can be traced back to a known sleep waveform, making the model's reasoning transparent to clinicians.

### Visualizations
- No figure needed for this section — keep it text-only on the poster (3-5 sentences max).
- Optional: A simple graphic showing the tension: "Accuracy vs. Interpretability" with this paper resolving the trade-off.

### Course Connection
- **Slides 10-14 (Post-WIMP medical applications):** Sleep monitoring is a clinical sensing application, analogous to rehabilitation and medical sensor examples in the course. The speech script discusses patients interacting with systems that track their physiological state and provide feedback — sleep staging is exactly this paradigm.

### Poster Hint
Keep this short. One paragraph max. The examiner cares about the MM-Systems connection more than general medical motivation. Key message: "sleep matters clinically + DL is accurate but opaque = this paper solves both."

---

## Section 2: The Domain Problem — Sleep Stage Scoring

### Text Draft

#### What Is Sleep Staging?

Sleep is organized into recurring cycles of distinct physiological states. The gold standard for measuring sleep is **polysomnography (PSG)** — simultaneous recording of brain activity (EEG), eye movements (EOG), and muscle tone (EMG). Trained technicians visually inspect these recordings in **30-second windows (epochs)** and assign each epoch one of five stages defined by the **American Academy of Sleep Medicine (AASM)**:

| Stage | Key EEG Signature | EOG | EMG | Clinical Meaning |
|-------|-------------------|-----|-----|------------------|
| **Wake (W)** | Alpha rhythm (8-13 Hz), eye blinks | Voluntary movements | Normal tone | Alert or drowsy wakefulness |
| **N1** | Low-amplitude mixed frequency, theta (4-7 Hz) | Slow rolling eye movements | Slightly reduced | Light sleep onset; most ambiguous stage |
| **N2** | Sleep spindles (12-15 Hz bursts), K-complexes | Absent | Reduced | Stable light sleep; ~50% of total sleep |
| **N3 (SWS)** | High-amplitude slow waves (delta, 0.5-4 Hz), >75 uV | Absent | Low | Deep/restorative sleep |
| **REM** | Low-amplitude mixed frequency (similar to N1) | Rapid conjugate eye movements | Atonia (near-zero) | Dreaming; memory consolidation |

A typical night contains 4-6 cycles of N1-N2-N3-N2-REM, each lasting ~90 minutes, with REM periods lengthening toward morning.

#### The AASM Scoring Rules

The AASM Manual (Berry et al., 2017) codifies precise waveform-recognition criteria that technicians follow:

- **Sleep spindle:** A burst of 11-16 Hz activity lasting >=0.5 seconds, with distinct crescendo-decrescendo morphology. Presence -> score N2 (unless criteria for N3 are met).
- **K-complex:** A well-delineated negative sharp wave immediately followed by a positive component, standing out from background EEG. Duration >=0.5 seconds. Presence -> score N2.
- **Slow waves:** Waves of frequency 0.5-2 Hz with peak-to-peak amplitude >75 uV, measured over frontal derivations. If >=20% of epoch -> score N3.
- **Rapid eye movements:** Conjugate, irregular, sharply-peaked movements. Combined with low chin EMG (atonia) -> score REM.

These rules are the "expert knowledge" that the paper infuses into its architecture.

#### Why Is This Hard?

- **Inter-rater agreement:** Even trained experts agree only ~80% overall. For **N1**, agreement drops to 52-60% — it is genuinely ambiguous because N1 EEG resembles both Wake and REM.
- **Scale:** A single overnight PSG produces ~1,000 epochs. A sleep lab processing 20 patients/night generates 20,000 epochs requiring manual review.
- **Subjectivity:** Borderline epochs (e.g., epoch with 18% slow waves — N2 or N3?) require judgment calls that vary between scorers.

### Visualizations
- **From AASM handbook or textbook:** A figure showing example EEG/EOG/EMG traces for each sleep stage (one row per stage). This is the single most useful educational figure. Check `project/papers/aasm-sleepschedule.pdf` for a clean version.
- **Hypnogram:** A classic sleep architecture diagram showing the cycling through stages across a night (time on x-axis, stages on y-axis).
- **Paper Figure 4:** Expert-scored hypnogram vs. single-epoch vs. multi-epoch network output. Shows both the problem AND the solution.

### Course Connection
- **Frame-based analysis (Lecture 5, Speech processing):** 30-second PSG epochs = exact analogue of speech analysis frames for MFCC extraction. Same stationarity assumption within fixed-length windows.
- **Unitizing (Slides 137-145):** The speech script defines "Interval Coding" as dividing recordings into fixed-length time intervals. The 30-second epoch is textbook interval coding.
- **Feature design philosophy (Lecture 3 — 10 groups of features / Laban Effort):** AASM rules convert clinical expert knowledge into computable signal features — same methodology as converting Laban movement theory into kinematic feature groups. Both operationalize domain expertise into measurable quantities. *(Already identified in project14-plan.md Step 3.)*

### Poster Hint
The stage table is poster-worthy. It immediately conveys the multimodal nature: **every stage requires information from multiple modalities**. You cannot distinguish N1 from REM using EEG alone (both are low-amplitude mixed-frequency) — you need EOG (eye movements) and EMG (atonia) to disambiguate. This is where you establish the paper as a multimodal system in the course sense.

---

## Section 3: The Technical Problem — Why DL Alone Isn't Enough

### Text Draft

Machine learning has been applied to automatic sleep staging for over two decades. Classical approaches — SVMs, random forests, HMMs — relied on hand-crafted features (spectral power in frequency bands, signal entropy, waveform detectors) designed by domain experts. These were interpretable but limited in accuracy.

Deep learning changed the equation. Starting with DeepSleepNet (Supratak et al., 2017), which used CNN+BiLSTM to learn directly from raw EEG, DL methods consistently outperformed classical approaches. By 2022, models like SleepTransformer and XSleepNet reached ~84-88% accuracy on standard benchmarks.

But accuracy alone is insufficient in clinical medicine. The stakes are high: sleep staging informs diagnosis of sleep apnea, narcolepsy, insomnia, and neurological disorders. A misclassified epoch can shift clinical metrics (e.g., REM latency, slow-wave sleep percentage) that drive treatment decisions. In this context:

1. **DL models are tools, not decision-makers.** The clinician must remain the authority. A model that says "this is N2" without explaining *why* forces the clinician to either blindly trust it or ignore it entirely — neither is acceptable.

2. **Black-box behavior breaks the clinical workflow.** Sleep technicians are trained to look for specific waveforms (spindles -> N2, slow waves -> N3, eye movements -> REM). If a model's reasoning cannot be expressed in these terms, it cannot be integrated into existing clinical practice.

3. **Trust requires verifiability.** Clinicians need to see that the model is "looking at the right things." Generic DL interpretability methods (Grad-CAM, saliency maps) produce heatmaps that don't map onto clinical vocabulary.

This is the gap the paper fills: **accurate AND interpretable, in clinically meaningful terms.** Expert knowledge is built directly into the network architecture — not as a post-hoc explanation, but as a structural constraint on what the network can learn.

### Visualizations
- **Comparison table (condensed from paper Table 6):**

| Method | Year | Channels | ACC (%) | kappa |
|--------|------|----------|---------|-------|
| DeepSleepNet | 2017 | EEG | 82.0 | 0.76 |
| SleepEEGNet | 2018 | EEG | 84.3 | 0.79 |
| SeqSleepNet-FT | 2019 | EEG-EOG | 84.3 | 0.78 |
| DeepSleepNet-FT | 2020 | EEG-EOG | 84.6 | 0.78 |
| **Proposed** | **2024** | **EEG-EOG** | **93.94** | **0.88** |

- Optional: A visual spectrum — "Classical ML: Interpretable but less accurate" vs. "Deep Learning: Accurate but opaque" vs. "This paper: Both."

### Course Connection
- **ML classification methods (Lecture 3):** The course covers kNN, HMM, SVM, and DL as classification approaches. This section positions the paper within that taxonomy: classical ML -> DL -> interpretable DL.
- **Prediction step in the multimodal pipeline (Slides 15-20):** Sleep staging is exactly the "prediction" step the speech script describes: *"Given the representation, you want to infer something: a gesture class, an emotional state, an action label."*

### Poster Hint
Keep shorter than Section 2. The comparison table is the strongest visual — it shows the problem (prior methods plateau ~84%) and the solution (proposed reaches ~94%) in one glance. The +9.3% gap is striking.

---

## Section 4: "Infusing Expert Knowledge" — What It Means and How It's Done

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

## Section 5: Architecture — Full Explanation

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

## Section 6: Data

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

## Section 7: Evaluation / Results

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

## Section 8: Interpretability — How the Goal Was Achieved

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

## Section 9: Conclusion, Impact, and Future Work

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
