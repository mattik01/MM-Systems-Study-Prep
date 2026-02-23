# Poster Content — Final Version
## Niknazar & Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
### IEEE TPAMI, vol. 46, no. 7, pp. 5044-5061, July 2024

---

## 1. Explainability in Deep Learning Matters

Deep learning has revolutionized many fields, in many cases drastically outperforming classical machine learning approaches. Yet these gains come at a cost: deep neural networks exhibit black-box behaviour. A model may achieve very high accuracy with high apparent confidence while offering no account of its reasoning. This poses a critical problem in high-stakes domains — medical applications being a prime example. Responsibility for decision-making must ultimately remain with trusted human experts, with software serving only as a tool to accelerate and reduce error in their work, never as an authority. Clinicians cannot responsibly act on opaque decisions, and regulations increasingly enforce standards of accountability for AI systems (EU AI Act 2024/1689, Art. 13 & Annex III; FDA, 2025, App. B).

Three key terms define this space:

- **Explainability:** A post-hoc account of why a model reached a given decision, typically expressed in summarized form.
- **Interpretability:** The degree to which a model's intrinsic properties are disclosed in a way that allows understanding of how decisions are made.
- **Transparency:** Achieved when intrinsic architectural properties produce human-readable explanations for a model's decisions without additional post-hoc analysis.

The paper under review proposes a transparent-by-design approach: the first layer of a CNN architecture is constrained to match established clinical vocabulary and expert decision-making criteria, making the model's representations directly interpretable. The network thereby offers clinically meaningful insight into its decisions as a structural property of the architecture itself.

This approach is demonstrated in the domain of automatic sleep stage scoring — a clinically significant classification task that demands both high performance and close alignment with expert knowledge.

---

## 2. The Sleep Stage Scoring Problem

Sleep is understood as a cycle of distinct physiological stages, each identifiable by characteristic signal patterns across brain activity (EEG), eye movements (EOG), and muscle tone (EMG). The American Academy of Sleep Medicine (AASM) defines five stages, each requiring its own multimodal evidence, and serves as the gold standard for sleep stage scoring. A typical night progresses through 4 to 6 such cycles, each lasting approximately 90 minutes, with REM periods lengthening towards the morning.

Analyzing sleep stages is clinically essential: it is a key basis for assessing sleep quality and for diagnosing not only sleep disorders but a broad range of conditions including depression and neurodegenerative disease.

| Stage | Key EEG Signature | EOG | EMG | Meaning |
|-------|-------------------|-----|-----|---------|
| **Wake** | Alpha (8-13 Hz) | Blinks (0.5-2 Hz); rapid eye movements | Variable; higher than sleep stages | Alert / drowsy |
| **N1** | Low-amplitude mixed, predominantly 4-7 Hz | Slow rolling eye movements | Slightly reduced | Light sleep onset |
| **N2** | Sleep spindles (11-16 Hz); K-complexes | Typically absent | Reduced | Stable light sleep (~50% of night) |
| **N3 (SWS)** | Slow waves, 0.5-2 Hz, >75 uV (>=20% of epoch) | Absent | Low | Deep / restorative |
| **REM** | Low-amplitude mixed (similar to N1); sawtooth waves | Rapid conjugate eye movements | Atonia (near-zero) | Dreaming / memory consolidation |

Domain experts score 30-second epochs by detecting characteristic waveforms in the signal, following the AASM Sleep Scoring Manual. This process is laborious and subjective — experts reach agreement on only around 80% of epochs on average; for N1, agreement drops to 52-60%.

Automatic scoring has therefore attracted sustained research interest. Since 2017, deep learning approaches (DeepSleepNet, XSleepNet) have achieved high accuracy on this task, reaching 82-88%. Yet these models inherit the black-box problem — their reasoning cannot be expressed in the waveform vocabulary sleep technicians are trained to use. The paper under review takes this further by incorporating interpretability alongside excellent predictive performance.

---

## 3. Expert Knowledge Infusion — The Gabor Kernels

As hinted at in the previous section, expert knowledge is infused into the network through a simple mechanism: using Gabor functions as the network's first convolutional layer, in place of the arbitrary learned weights of a standard 1D convolutional filter.

A Gabor function is a cosine wave modulated by a Gaussian envelope, with only three learnable parameters: frequency, bandwidth, and time offset within a two-second window. It models localized oscillations in a temporal-frequency signal and can effectively take the form of the characteristic waveforms experts look for when scoring sleep stages — slow waves, theta waves, and spindles (see Table X). The reduced number of EOG filters (8 vs. 32 for EEG) reflects the simpler spectral structure of eye-movement signals.

What is the consequence? Expert knowledge is not inserted as explicit rules or auxiliary loss terms; rather, it emerges through the informational bottleneck imposed by the Gabor functional form. The model is forced to look at the signal in the same way an expert would and to base all further reasoning on that representation. Not even the specific waveforms are imposed — as usual, the model learns them through training — but the authors found that several kernels converged to the form of known sleep-stage indicators from the AASM scoring manual (see Table Y).

---

## 4. Architecture

**Input:** One 30-second PSG epoch (EEG + EOG channels at 100 Hz), Z-score normalized to standardize amplitude across recordings.

The system has two levels, each corresponding to a different scale of clinical reasoning:

### Level 1: Single-Epoch Network (within one 30-second window)

A CNN architecture whose first filter layer is constrained to learning Gabor kernels (explained in detail in the next section). It applies 32 filters to the EEG channel and 8 to the EOG channel, followed by a mixing layer and then four standard CNN blocks. The network ends with fully connected layers including dropout, producing five outputs — one per sleep stage. This mirrors an expert evaluating a single epoch at a time.

### Level 2: Multi-Epoch Network (across consecutive epochs)

For windows of nine consecutive epochs, the outputs of the Level 1 network are fed into two Bi-LSTM layers (capturing both preceding and following epochs), followed by a fully connected layer that collapses the temporal dimension into a five-class final prediction for the center epoch. This allows the model to consider context around each epoch, respect the cyclical nature of sleep stages, and avoid noise and single-epoch classification errors (see comparison in Figure 2 above).

---

## 5. Evaluation

The system was evaluated on publicly available polysomnography datasets. The primary dataset is Sleep-EDF Expanded, comprising 78 subjects with two overnight recording sessions each. DREAMS, with a slightly different recording setup and scoring standard, was used for independent cross-dataset validation. Probabilistic mini-batch sampling was employed to address the inherent class imbalance in sleep stages during training.

The model is evaluated under four cross-validation protocols on the Sleep-EDF data: night-holdout (train on night 1, test on night 2), subject-holdout (unseen patients), record-holdout (fixed split for interpretability experiments), and leave-one-out on EDF-20 (direct baseline comparison with previous studies). The validation also used a larger test ratio (0.2) compared to most previous studies, indicating more difficult testing conditions.

(Table of Comparison [1])

A noticeable gap in both accuracy and Cohen's kappa was achieved over previous methods. For the DREAMS dataset (table not presented), the gap rises to +9.3% accuracy and +0.10 kappa — quite impressive in a mature benchmark.

Ablation of each component's contribution revealed that the Gabor layer provides the largest single gain (+10% over a standard CNN), followed by the multi-epoch network (+6.2%, correcting implausible stage transitions) and EOG (+1.3%, most impactful for N1 and REM).

### 5.7 Interpretability Validation — The Four-Level Framework

The authors introduce a four-level interpretability framework that illustrates how the proposed intrinsic method is far superior to any post-hoc approach in terms of interpretability.

**Level 1 — What did the Gabor kernels learn?** Visualizations of the optimized kernels show that they cluster around clinically relevant frequency bands. The model independently rediscovers and uses filters that experts already know matter.

**Level 2 — Which kernels matter for which stages?** Using the Effective Functional Effect (EFF) metric, the authors measure each kernel's contribution to each stage's classification and find, once again, consistency between learned waveforms and their corresponding sleep stages.

**Level 3 — Which modality drives each stage?** A potential weak point of the paper is its limited explanation of how Gabor kernels model EOG indicators effectively, given the lack of prior-work foundation comparable to EEG and the use of only 8 kernels for that channel instead of 32. Nevertheless, by measuring the impact of both channels on different stages' classification, significantly higher EOG dependence is detected in stages N1 and REM — which is fully consistent with AASM criteria and the expected involvement of eye movements. The model weights modalities exactly as expected.

**Level 4 — Waveform detection within a single epoch (granular level).** The model reveals exactly when, within any given epoch, a kernel activates — which waveform was detected, at what point in time, and how much it contributed. This is something a trained clinician can directly understand as an explanation of the model's decision and verify against the raw signal.

---

## 6. Conclusion and Outlook

The paper demonstrates how a simple constraint — forcing the model to reason through the same vocabulary that experts use — not only makes the system immediately more interpretable and verifiable by clinicians, but simultaneously improves generalization. The assumed accuracy–interpretability trade-off is shown to be false: an inductive bias grounded in established domain knowledge achieves both at once.

**Limitations:** EMG is omitted despite the AASM requiring it for REM scoring; only the first layer is transparent (deeper layers rely on indirect EFF attribution); Gabor functions may miss non-oscillatory transients and certain artefacts.

**Future directions:** Additional modalities (EMG, ECG), negative gradient analysis, and extension to clinical populations with sleep pathology.

---

## 7. Course Connections (MM-Systems)

| Course Concept | Slide Reference | Paper Connection |
|---------------|-----------------|------------------|
| Multimodal processing pipeline | 15-20 | The paper IS this pipeline: sensors -> features -> classification -> labels |
| Four-layer feature hierarchy | 53-65 | Architecture maps directly: raw signals -> Gabor activations -> CNN features -> stage labels |
| Early fusion | 6-8 | Mixing layer combines EEG + EOG features before classification |
| CARE / Complementarity | 35-41 | EEG + EOG are complementary; ablation confirms neither suffices alone |
| Frame-based analysis / Unitizing | 137-145 | 30-second epochs = interval coding, same as MFCC speech frames |
| Feature design from domain knowledge | Laban Effort, Lecture 3 | AASM rules -> Gabor features, same as Laban theory -> movement features |
| CNN-LSTM hybrid | 2034-2061 | CNN for within-epoch patterns + LSTM for across-epoch temporal dynamics |
| Hand-crafted vs. learned features | 25-26 | Gabor = hybrid: hand-crafted functional form, learned parameters |
| ML classification & metrics | Lecture 3 | kNN/SVM/DL taxonomy; Cohen's kappa for imbalanced classes |
| Robustness & disambiguation | 3648-3654 | EOG disambiguates N1/REM from N2/SWS |
| Five MM-ML challenges | 3640-3690 | Addresses Representation, Fusion, and Co-learning challenges |
| Post-WIMP medical applications | 10-14 | Sleep staging as clinical sensing application |

---

## References

[1] M. Niknazar and S. C. Mednick, "A multi-level interpretable sleep stage scoring system by infusing experts' knowledge into a deep network architecture," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 7, pp. 5044–5061, Jul. 2024.

[2] European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act)," *Official Journal of the European Union*, L 2024/1689, Jul. 2024.

[3] U.S. Food and Drug Administration, "Artificial intelligence-enabled device software functions: Lifecycle management and marketing submission recommendations," Draft Guidance, Docket No. FDA-2024-D-4488, Jan. 2025.

[4] S. Ali et al., "Explainable artificial intelligence (XAI): What we know and what is left to attain trustworthy artificial intelligence," *Inf. Fusion*, vol. 99, art. 101805, Nov. 2023.

[5] R. B. Berry et al., *The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications*, Version 2.4. Darien, IL, USA: American Academy of Sleep Medicine, 2017.

[6] A. Supratak, H. Dong, C. Wu, and Y. Guo, "DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 11, pp. 1998–2008, Nov. 2017.

[7] S. Mousavi, F. Afghah, and U. R. Acharya, "SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach," *PLOS ONE*, vol. 14, no. 5, art. e0216456, May 2019.

[8] H. Phan, O. Y. Chen, M. C. Tran, P. Koch, A. Mertins, and M. De Vos, "XSleepNet: Multi-view sequential model for automatic sleep staging," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 9, pp. 5903–5915, Sep. 2022.

[9] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 2, pp. 423–443, Feb. 2019.

[10] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, "Analysis of a sleep-dependent neuronal feedback loop: The Sleep-EDF database," *IEEE Trans. Biomed. Eng.*, vol. 47, no. 9, pp. 1185–1194, Sep. 2000.

[11] S. Devuyst, T. Dutoit, P. Stenuit, and M. Kerkhofs, "The DREAMS databases and assessment algorithm," in *Proc. 33rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)*, Boston, MA, USA, 2011, pp. 4343–4346.
