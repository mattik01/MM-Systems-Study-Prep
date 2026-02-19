# Paper Acquisition List — Project 14

Papers to obtain and read before/alongside the main paper.
Check off each as you acquire the PDF.

---

## Main Paper (required)

- [ ] **Niknazar & Mednick (2024)**
  H. Niknazar and S. C. Mednick, "A Multi-Level Interpretable Sleep Stage Scoring System by Infusing Experts' Knowledge Into a Deep Network Architecture"
  *IEEE TPAMI*, vol. 46, no. 7, pp. 5044–5061, July 2024
  DOI (probable): `10.1109/TPAMI.2024.3364430`
  Source: IEEE Xplore / UniGe library VPN / Sara Mednick lab page (UC San Diego) / arXiv preprint

---

## Essential Background

- [ ] **AASM Sleep Scoring Manual**
  R. B. Berry et al., *AASM Manual for the Scoring of Sleep and Associated Events*, Rules, Terminology and Technical Specifications, v2.6
  Available free: https://aasm.org/clinical-resources/scoring-manual/
  *Why*: The paper's "expert knowledge" IS this manual — must understand it to evaluate what gets injected.

- [ ] **Rechtschaffen & Kales (1968)** *(short, historical context)*
  A. Rechtschaffen and A. Kales (Eds.), "A Manual of Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human Subjects"
  Public domain — search "Rechtschaffen Kales 1968 sleep scoring PDF"
  *Why*: Original R&K rules; understand to see how AASM updated them.

---

## Direct DL Baselines (appear in paper's comparison table)

- [ ] **DeepSleepNet — Supratak et al. (2017)**
  A. Supratak, H. Dong, C. Wu, and Y. Guo, "DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG"
  *IEEE Trans. Neural Systems and Rehabilitation Engineering (TNSRE)*, vol. 25, no. 11, pp. 1998–2008, 2017
  DOI: `10.1109/TNSRE.2017.2721116`
  *Why*: CNN+BiLSTM baseline; likely in the results table; key reference point for "what came before."

- [ ] **SeqSleepNet — Phan et al. (2019)**
  H. Phan, F. Andreotti, N. Cooray, O. Y. Chén, and M. De Vos, "SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging"
  *IEEE Trans. Neural Systems and Rehabilitation Engineering (TNNLS)*, 2019
  DOI: `10.1109/TNSRE.2019.2936800`
  *Why*: Sequence-to-sequence DL approach; another key baseline.

- [ ] **XSleepNet — Phan et al. (2021)**
  H. Phan et al., "XSleepNet: Multi-View Sequential Model for Automatic Sleep Staging"
  *IEEE Trans. Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 44, no. 9, pp. 5903–5915, Sept. 2022
  DOI: `10.1109/TPAMI.2021.3070057`
  *Why*: Joint spectral+temporal view model; high-performing baseline in same journal.

- [ ] **SleepTransformer — Phan et al. (2022)**
  H. Phan et al., "SleepTransformer: Automatic Sleep Staging with Interpretability and Uncertainty Quantification"
  *IEEE Trans. Biomedical Engineering*, vol. 69, no. 8, pp. 2456–2467, 2022
  DOI: `10.1109/TBME.2022.3147187`
  *Why*: Transformer-based; also claims interpretability — good to compare approaches.

---

## Interpretability in DL (context)

- [ ] **Zhang & Zhu (2018)** *(survey, skim suffices)*
  Q. Zhang and S.-C. Zhu, "Visual Interpretability for Deep Learning: A Survey"
  *Frontiers of Information Technology & Electronic Engineering*, vol. 19, no. 1, pp. 27–39, 2018
  *Why*: Quick survey of what "interpretable DL" means — puts the paper's claims in context.

---

## Input Format Context

- [ ] **Chambon et al. (2018)**
  S. Chambon, M. N. Galtier, P. J. Arnal, G. Wainrib, and A. Gramfort, "A Deep Learning Architecture for Temporal Sleep Stage Classification Using Multivariate and Multimodal Time Series"
  *IEEE Trans. Neural Systems and Rehabilitation Engineering (TNSRE)*, vol. 26, no. 4, pp. 758–769, Apr. 2018
  DOI: `10.1109/TNSRE.2018.2813138`
  *Why*: Establishes the multimodal EEG/EOG/EMG input format used in the paper.

---

## Nice-to-Have (if time permits)

- [ ] **U-Sleep — Perslev et al. (2021)**
  M. Perslev et al., "U-Sleep: Resilient High-Frequency Sleep Staging"
  *npj Digital Medicine*, vol. 4, article 72, 2021
  *Why*: U-Net-based; multi-dataset training; shows the scale of modern sleep DL work.

- [ ] **GraphSleepNet — Jia et al. (2020)**
  Z. Jia et al., "GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification"
  *IJCAI 2020*
  *Why*: Graph convolution over EEG channels; alternative architectural approach.

---

## Notes
- Prioritise: Main paper → AASM manual → DeepSleepNet → SeqSleepNet → SleepTransformer
- R&K manual is historical context only, low priority unless you want the full evolution story
- Chambon 2018 is helpful if the main paper references it heavily for input format justification
