# Project 14 — Critical Reading Plan
## Niknazar & Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
### *IEEE TPAMI*, vol. 46, no. 7, pp. 5044–5061, July 2024

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| Structured study notes (Markdown) | `project/study-notes.md` | [ ] |
| A0 poster (PowerPoint) | `project/project14-poster.pptx` | [ ] |
| 10-min presentation (oral) | — rehearsal only — | [ ] |

**Submission deadline:** at least one week before the exam date.

---

## Step 1: Acquire the Paper

- Search IEEE Xplore: `10.1109/TPAMI.2024.3374072` (or search title)
- Search Google Scholar: "Niknazar Mednick multi-level interpretable sleep stage scoring TPAMI 2024"
- Download full PDF + supplementary material
- Use university library VPN if paywalled (University of Genoa / DIBRIS access)

**What to look for when reading:**
- Full list of input modalities (EEG channels, EOG, EMG)
- Architecture diagram (multi-level = how many levels? what is each level?)
- Expert knowledge injection mechanism (attention? regularization? rule-based constraints?)
- Training datasets (SHHS? Sleep-EDF? MASS? — note sizes)
- Key results table (accuracy, F1, Cohen's kappa vs. baselines)
- Interpretability visualizations (activation maps, attention weights)

---

## Step 2: Background Reading

Read in this order before or alongside the main paper:

### Essential (must read)
1. **AASM Sleep Scoring Manual** (Berry et al., 2012/2018) — the clinical standard for sleep staging (N1, N2, N3, REM, Wake). Available free at aasm.org. The paper's "expert knowledge" is grounded here.
2. **Rechtschaffen & Kales (1968)** — original R&K sleep staging rules; understand to appreciate how AASM evolved. Short read.

### Directly related DL methods (skim abstracts + results)
3. **Supratak et al., "DeepSleepNet" (IEEE TNSRE, 2017)** — CNN+BiLSTM baseline; likely in the paper's comparison table
4. **Phan et al., "SeqSleepNet" (IEEE TNNLS, 2019)** — sequence-to-sequence DL approach; context for what this paper improves on
5. **Chambon et al. (IEEE TNSRE, 2018)** — establishes multimodal EEG/EOG/EMG input format used in this paper

### Interpretability context
6. **Zhang & Zhu, "Visual Interpretability for Deep Learning: a Survey" (Frontiers IT&EE, 2018)** — what "interpretable" means in DL context

### From the paper itself
7. Scan the paper's **reference list** — find the 2–3 most-cited papers and skim their abstracts.

---

## Step 3: Map to Course Content

| Course Content (from speech script) | Relevance to Paper |
|---|---|
| **Slides 15–20**: Multimodal processing pipeline (input → representation → prediction → mapping → output) | The paper IS this pipeline: EEG/EOG/EMG sensors → feature extraction → DL classifier → sleep stage label |
| **Four-layer hierarchy** (Lecture 3): physical signals → low-level → mid-level → concepts | Paper's "multi-level" architecture mirrors this: raw EEG → time/freq features → expert-rule features → sleep stage label |
| **Slides 10–14**: Post-WIMP examples (rehabilitation, medical sensors) | Sleep monitoring is a medical sensing context, analogous to rehabilitation examples |
| **Sensory modalities** (Slides 21–24) | EEG + EOG + EMG = three distinct physiological modalities → this IS a multimodal system |
| **CARE / Redundancy / Complementarity** (Lecture 2) | Multi-channel fusion: EOG and EEG carry complementary information about sleep stages |
| **Early/late/hybrid fusion** (Lecture 3) | Paper uses specific fusion strategy across modalities — identify which type |
| **10 groups of low-level features** / Laban Effort (Lecture 3) | Analogous feature design philosophy: expert domain knowledge encoded as computable signal features |
| **Frame-based analysis, MFCC extraction** (Lecture 5, speech) | EEG analyzed in 30-second fixed-length epochs = exactly like speech frames |
| **ML classification** (kNN, HMM, SVM, DL — Lecture 3) | Paper uses deep network; comparing to classical approaches is framing context |

**Use this mapping in the poster's introduction/discussion** to anchor the paper within the MM-Systems course.

---

## Step 4: Deep-Read Questions (structured notes guide)

Use these questions to focus your reading — answers become poster content:

### Objectives
- [ ] What is the clinical problem? (manual staging: expensive, subjective, expert-dependent)
- [ ] What gap does the paper fill? (existing DL = black box, clinicians cannot trust/correct it)
- [ ] What is the claimed contribution? (multi-level architecture: accurate + interpretable via expert rule injection)

### Methodology
- [ ] What input modalities are used? (list all EEG channels, EOG, EMG)
- [ ] How are 30-second epochs created? (windowing strategy)
- [ ] What is the "multi-level" structure? (map each level to the four-layer hierarchy)
- [ ] How exactly is expert knowledge "infused"? (attention mechanisms? rule-based regularization? architectural inductive bias?)
- [ ] What DL components are used? (CNN, LSTM, attention, transformer?)
- [ ] What datasets? (SHHS, Sleep-EDF, MASS — sizes, demographics)

### Results
- [ ] Metrics used? (accuracy, F1, Cohen's kappa)
- [ ] How does it compare to baselines? (table: SeqSleepNet, DeepSleepNet, etc.)
- [ ] Which sleep stage is hardest? (usually N1 — discuss why)
- [ ] What do interpretability results show? (attention maps, rule activation patterns)

### Discussion
- [ ] What are the limitations? (dataset bias, generalization, clinical deployment)
- [ ] What future work is proposed?

---

## Step 5: Markdown Study Notes

**File**: `project/study-notes.md`

Sections:
1. Abstract / One-paragraph summary
2. Problem Statement
3. Architecture (with text diagram of each level)
4. Datasets & Evaluation Protocol
5. Results (reproduce key table and F1 per stage)
6. Discussion (strengths, limitations, future work)
7. Course Connections (multimodal pipeline, four-layer hierarchy)
8. Exam Preparation (key questions + answers)

Include:
- Key equations from the paper (loss function, attention formula, expert constraint formulation)
- The course-slide mapping table (Step 3 above)
- Critical analysis notes (convincing vs. questionable claims)

> **Note**: Keep notes in Markdown — pull content directly into the PowerPoint poster when building it.

---

## Step 6: A0 Poster Design

**File**: `project/project14-poster.pptx`

**Size**: A0 = 841mm × 1189mm portrait

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  TITLE BAR: Paper title | Authors | IEEE TPAMI 2024 | Name  │
├──────────────────┬──────────────────┬───────────────────────┤
│  MOTIVATION &    │   METHODOLOGY    │    KEY RESULTS        │
│  OBJECTIVES      │                  │                       │
│                  │ • Architecture   │ • Accuracy table vs.  │
│ • Sleep stage    │   diagram        │   baselines           │
│   problem        │ • Multi-level    │ • Per-stage F1 bars   │
│ • Clinical gap   │   structure      │ • Interpretability    │
│ • Why DL fails   │ • Expert rule    │   visualization       │
│ • Course:        │   injection      │                       │
│   MM pipeline    │   mechanism      │                       │
│   connection     │ • Datasets       │                       │
├──────────────────┴──────────────────┴───────────────────────┤
│  DISCUSSION: Strengths | Limitations | Future Work          │
│  + MM-Systems course connection (multimodal pipeline)       │
└─────────────────────────────────────────────────────────────┘
```

**Typography (A0 readability)**:
- Title: ~72pt
- Section headers: ~36pt
- Body text: ~24pt minimum

**Critical elements (must be on poster)**:
- [ ] Architecture diagram (central element)
- [ ] Results table or figure
- [ ] Explicit course pipeline connection
- [ ] All four sections: Objectives, Methodology, Results, Discussion

---

## Step 7: 10-Minute Presentation Structure

| Segment | Content | Time |
|---|---|---|
| Problem context | Why sleep staging matters; what makes it hard | ~1.5 min |
| Paper objectives | Clinical gap; claimed contribution | ~1 min |
| Architecture walkthrough | Multi-level design, each level explained | ~3 min |
| Results | Accuracy comparison + interpretability example | ~2 min |
| Critical analysis | What's convincing; what's a limitation | ~1.5 min |
| Course connection | Multimodal pipeline; four-layer hierarchy | ~1 min |

### Anticipated exam questions — prepare answers
1. "What exactly is novel compared to DeepSleepNet?"
2. "How do they inject expert knowledge — what is the exact mechanism?"
3. "Which modality contributes most to classification performance?"
4. "How is this system a multimodal system in the MM-Systems sense?"
5. "What does 'interpretable' mean here, and for whom?"

---

## Step 8: Execution Checklist

- [ ] 1. Download paper PDF (+ supplementary)
- [ ] 2. Read AASM rules summary (30 min) + skim DeepSleepNet/SeqSleepNet abstracts (15 min)
- [ ] 3. Deep-read paper using Step 4 questions (3–4 hours)
- [ ] 4. Write Markdown study notes while reading (`project/study-notes.md`)
- [ ] 5. Build poster in PowerPoint — architecture diagram is central element (`project/project14-poster.pptx`)
- [ ] 6. Rehearse 10-minute presentation aloud against poster (1 hour)

---

## Verification Checklist (before submission)

- [ ] Poster covers all four required sections: Objectives, Methodology, Results, Discussion
- [ ] Architecture diagram present and labeled on poster
- [ ] At least one results table or figure on poster
- [ ] Course connection (multimodal pipeline / four-layer hierarchy) explicitly stated on poster
- [ ] LaTeX notes contain all key equations from paper
- [ ] Presentation fits within 10 minutes when rehearsed aloud
- [ ] Submitted at least one week before exam

---

## Reference

> H. Niknazar and S. C. Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 7, pp. 5044–5061, July 2024.
