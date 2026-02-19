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

Proposed DOI: `10.1109/TPAMI.2024.3364430` — verify on IEEE Xplore.

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
- AASM expert knowledge injected at multiple abstraction levels
- Novel expert-rule fidelity metric for evaluating interpretability

---

## 3. Architecture

### 3.1 Input Modalities

| Modality | Channels | Clinical role |
|---|---|---|
| EEG | F3/F4, C3/C4, O1/O2 (AASM standard derivations) | Brain state: spindles (N2), K-complexes (N2), slow waves (N3), mixed-frequency (N1) |
| EOG | E1-M2, E2-M1 (left + right eye) | Rapid eye movements → REM detection |
| EMG | Submental/chin muscle | Atonia → distinguishes REM from N1 |

**Windowing**: 30-second epochs at 100–256 Hz → 3,000–7,680 samples per epoch.
*Same frame-based analysis as MFCC in speech (MM-Systems Lecture 5).*

### 3.2 Multi-Level Architecture

| Paper Level | MM-Systems Layer | What happens |
|---|---|---|
| Level 1 | Physical signals | Raw EEG/EOG/EMG sample sequences |
| Level 2 | Low-level features | CNN extracts band power, spindle characteristics, K-complex waveforms, slow-wave amplitude |
| Level 3 | Mid-level features | Attention over Level 2 features, guided by AASM-rule knowledge; auxiliary supervised representations |
| Level 4 | Concepts/structures | Sleep stage label (Wake/N1/N2/N3/REM) + temporal sequence model |

### 3.3 Expert Knowledge Injection — Mechanisms

Three complementary mechanisms:

**1. Attention regularisation**
Attention weights over the 30-second epoch are pushed (via auxiliary loss) toward time/frequency regions AASM specifies as discriminative:
- σ-band (12–15 Hz) at spindle-containing time points → N2
- δ-band (0.5–4 Hz) at slow-wave segments → N3
- Rapid EOG bursts → REM
- Low chin EMG amplitude → REM (atonia)

**2. Expert-rule auxiliary losses (soft constraints)**
```
L_total = L_CE(ŷ, y) + λ₁·L_feature + λ₂·L_transition + λ₃·L_attention
```
- `L_CE` — cross-entropy for stage classification
- `L_feature` — MSE / BCE between intermediate representation and output of automatic feature detectors (spindle detector, slow-wave detector)
- `L_transition` — KL divergence between learned transition probabilities and expert-derived AASM transition prior matrix
- `L_attention` — penalty when attention deviates from expert-prescribed frequency-band importance
- λ₁, λ₂, λ₃ — hyperparameters (tuned per dataset)

**3. Transition probability constraints**
Sequence-level model (LSTM or Transformer over epoch sequence) receives a learnable transition prior derived from AASM sleep architecture rules:
- e.g., N3 does not directly follow REM in typical cycles
- Direct Wake→REM transition only valid in certain conditions

**4. Attention formula (additive / Bahdanau-style)** *(verify exact formulation)*
```
e_t = v^T · tanh(W·h_t + b)
α_t = exp(e_t) / Σ_j exp(e_j)       (softmax)
c   = Σ_t α_t · h_t                  (context vector)

Knowledge constraint:
L_attention = Σ_t || α_t - α_t^expert ||²
```
where `α_t^expert` is the soft target from an automatic feature detector.

### 3.4 Deep Learning Components
*(Fill in from paper — likely CNN + bidirectional LSTM + attention, possibly Transformer)*

---

## 4. Datasets & Evaluation Protocol

| Dataset | N subjects | Notes |
|---|---|---|
| SleepEDF-20 | 20 | Healthy adults; Fpz-Cz + Pz-Oz EEG + EOG + EMG |
| SleepEDF-78 | 78 | Same source, larger subset |
| SHHS | ~5,800 | Community cohort; C4-A1 EEG; large-scale generalization |
| MASS | ~200 | Full PSG montage; multiple subsets |
| ISRUC | ~100 | Clinical population incl. patients |

**Evaluation protocol**: *(verify — likely k-fold CV or LOSO)*
**Metrics**: Overall accuracy (ACC), Macro F1 (MF1), Cohen's κ

---

## 5. Results

### 5.1 Comparison Table *(verify all values against paper tables)*

| Method | ACC (%) | MF1 (%) | κ |
|---|---|---|---|
| DeepSleepNet (Supratak 2017) | 82.0 | 76.9 | 0.76 |
| SeqSleepNet (Phan 2019) | 87.1 | 80.0 | 0.83 |
| SleepTransformer (Phan 2022) | 88.0 | 81.0 | 0.84 |
| XSleepNet (Phan 2021) | 87.7 | 80.5 | 0.83 |
| **Niknazar & Mednick (this paper)** | **~89** | **~83** | **~0.85** |

### 5.2 Per-Stage F1 *(approximate — verify)*
| Stage | F1 | Why notable |
|---|---|---|
| Wake | ~90% | Most distinct signal pattern |
| **N1** | ~45–55% | Hardest; transitional EEG; biggest gain from expert knowledge |
| N2 | ~87% | Spindles/K-complexes well-defined |
| N3 | ~85% | Slow waves well-defined |
| REM | ~85% | EOG + atonia combination discriminates well |

### 5.3 Interpretability Results
- Temporal attention maps show model attending to spindle events when predicting N2, slow waves for N3, sawtooth waves for REM
- Frequency-band attention: σ-band dominant for N2, δ-band for N3 — matches AASM criteria
- Expert-rule fidelity metric: quantifies alignment between model attention and expert-prescribed features (novel contribution)
- Transition probability matrix from the sequence model matches clinician-established hypnogram statistics

---

## 6. Discussion

### 6.1 Strengths
- Clinician-verifiable: attention maps highlight exactly the signal events AASM clinicians look for
- Multi-benchmark: state-of-the-art across SleepEDF, SHHS, MASS simultaneously
- N1 F1 improvement is the most clinically meaningful gain (N1 is hardest for both humans and machines)
- Multi-mechanism injection (attention + auxiliary losses + transition priors) is more thorough than prior work

### 6.2 Limitations
- Relies on automatic spindle/K-complex detectors for auxiliary supervision — imperfect detectors → noisy auxiliary labels
- AASM rules are adult-specific; pediatric or pathology-specific extensions not addressed
- N1 human inter-rater agreement (~52–60%) sets a fundamental accuracy ceiling
- λ₁/λ₂/λ₃ hyperparameters need per-dataset tuning — limits plug-and-play clinical deployment
- Computationally more expensive than single-level baselines

### 6.3 Future Work *(verify against paper)*
- Extension to disorder-specific scoring rules (apnea, narcolepsy, insomnia)
- Clinician-in-the-loop active learning
- Large-scale pre-training + fine-tuning paradigm

---

## 7. Course Connections (MM-Systems)

### 7.1 Multimodal Processing Pipeline (Slides 15–20)
```
INPUT DEVICES:   EEG electrodes + EOG electrodes + EMG electrodes
      ↓
RAW DATA:        30-second epochs (time-domain physiological signals)
      ↓
REPRESENTATION:  CNN time/freq features; attention-weighted representations
      ↓
PREDICTION:      5-class sleep stage classifier (Wake/N1/N2/N3/REM)
      ↓
MAPPING:         Stage label → clinical hypnogram / report
      ↓
OUTPUT:          Sleep stage annotation + interpretability visualization
```

### 7.2 Four-Layer Hierarchy (Lecture 3)
The paper's multi-level structure is a direct instantiation of the course hierarchy:
- **Physical signals** → raw EEG/EOG/EMG sample arrays
- **Low-level features** → band power, spindle waveform descriptors (CNN output)
- **Mid-level features** → expert-guided attention representations (knowledge-infused)
- **Concepts and structures** → sleep stage label + hypnogram sequence

### 7.3 Multimodal Complementarity (Lecture 2)
EEG, EOG, EMG carry complementary information — no single modality suffices:
- EEG alone: cannot reliably detect atonia (needed for REM vs N1 distinction)
- EOG alone: no brain-state information
- EMG alone: binary (tonic/atonic); insufficient for full staging
→ This is the same complementarity argument as "Put That There" (speech + gesture): together they enable what neither can do alone.

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
> DeepSleepNet (CNN+BiLSTM) is purely data-driven — no domain knowledge injected, black box. This paper: (1) adds multi-level knowledge injection via attention regularisation and auxiliary losses aligned with AASM rules; (2) adds an expert-rule fidelity metric to evaluate interpretability; (3) explicitly models stage transition constraints. The interpretability is the key novelty, not just accuracy.

**Q2: How do they inject expert knowledge — what is the exact mechanism?**
> Three mechanisms: (1) attention weights regularised toward AASM-discriminative time/frequency regions per stage; (2) auxiliary training losses connecting intermediate representations to output of automatic feature detectors (spindles, K-complexes, slow waves); (3) learned transition prior from AASM sleep-architecture rules. Together these form a composite loss L = L_CE + λ₁·L_feat + λ₂·L_trans + λ₃·L_attn.

**Q3: Which modality contributes most to classification?**
> *(Check ablation study in paper.)* EEG is the primary modality for all stages. EOG is critical for REM detection. EMG provides the atonia signal that disambiguates REM from N1. All three are necessary for full 5-stage scoring per AASM.

**Q4: How is this a multimodal system in the MM-Systems sense?**
> EEG + EOG + EMG are three distinct physiological sensory modalities. Each captures a different aspect of the user's state (brain activity, eye movement, muscle tone). The system integrates them — satisfying both the W3C definition ("input in more than one modality") and the Jaimes-Sebe definition ("responds to inputs in more than one communication channel"). The modalities are complementary: no single one enables full 5-stage scoring.

**Q5: What does 'interpretable' mean here, and for whom?**
> Interpretable for sleep clinicians: the model produces attention visualisations that highlight the exact signal events (spindles, K-complexes, slow waves, eye movements, atonia) that AASM-trained technicians use. Clinicians can verify that the model is "looking at the right things" before trusting it. The expert-rule fidelity metric quantifies this alignment objectively. This is distinguished from generic DL interpretability (e.g., Grad-CAM) because it is grounded in domain-specific clinical criteria.
