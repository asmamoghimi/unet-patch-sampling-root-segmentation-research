# Data Management Plan
**Researcher:** Asma Moghimi  
**Project:** Beyond Random Sampling: Improving U-Net Performance on Noisy Plant Root Images  
**Institution:** Breda University of Applied Sciences (BUas) — Applied Data Science and AI  
**Project Period:** April 16, 2026 – June 8, 2026  
**Version:** 1.0 | June 8, 2026

---

## 1. Data Collection and Analysis

### 1.1 Data Source
The dataset was not collected by the researcher. It consists of 492 annotated greyscale images of *Arabidopsis thaliana* root systems, collected by the **Hades in-vitro phenotyping system** at the **Netherlands Plant Eco-phenotyping Centre (NPEC)**. Each petri dish was sown with five seeds and photographed daily as roots developed. Each image is paired with a binary segmentation mask in which root pixels are labelled as foreground and background pixels as background.

The dataset was provided to BUas for educational and research purposes and spans three academic year cohorts: Y2B_23 (115 images), Y2B_24 (314 images), and Y2B_25 (63 images).

### 1.2 Data Format and Volume
- **Format:** Greyscale PNG images and binary PNG masks
- **Resolution:** Uniform after preprocessing (resized to consistent dimensions)
- **Total size:** 492 image–mask pairs (~moderate file size, stored on BUas GPU server)
- **No proprietary formats used** — all files are standard PNG

### 1.3 Data Processing and Analysis
The following processing steps were applied:

1. **Preprocessing:** Non-target images (e.g. potato dish images) were identified and excluded. All images and masks were converted to greyscale PNG. Masks were binarised at a threshold of 127. Pixel values were normalised to [0, 1] by dividing by 255.
2. **Data splitting:** A stratified 70/15/15 train/validation/test split was applied using a fixed random seed of 42 to ensure reproducibility and proportional cohort representation.
3. **Patch extraction:** 256×256 pixel patches were extracted from training images using three strategies: random sampling (baseline), weighted sampling (H1), and hard-example sampling (H2).
4. **Augmentation:** Random horizontal and vertical flipping was applied to training patches only, using a shared seed to preserve spatial correspondence between image and mask.
5. **Model training:** Three U-Net models were trained — one per sampling strategy — using identical hyperparameters (Adam optimiser, lr=0.001, batch size=8, early stopping patience=10).
6. **Evaluation:** Dice Similarity Coefficient and Intersection over Union (IoU) were computed on the held-out test set. Statistical analysis used the Wilcoxon signed-rank test, Cohen's d effect size, and 95% bootstrap confidence intervals (1,000 iterations).

All analysis code is documented in `Asmamoghimi_research_notebook.ipynb` and `EDA_Asmamoghimi.ipynb`. A fixed random seed of 42 is applied throughout to ensure full reproducibility.

---

## 2. Data Storage and Protection

### 2.1 Storage Location
All data is stored exclusively on the **BUas GPU server** at `/home/y2b/`. No data is stored on personal devices, personal cloud storage (e.g. Google Drive, Dropbox), or any external systems.

### 2.2 Data Integrity
- The **original raw dataset is never modified**. All preprocessing steps produce copies only, stored in a separate directory.
- A **fixed random seed (42)** is used for all stochastic operations, ensuring that results can be reproduced exactly from the original data.
- **Model checkpoints** are saved at each validation improvement during training. Only the best checkpoint per experiment is retained to avoid unnecessary storage use.

### 2.3 Access Control
Access to the data is restricted to:
- The researcher (Asma Moghimi)
- BUas supervisory and technical staff with legitimate access to the GPU server

No data is shared with third parties. The dataset is owned by BUas and is used strictly within the scope of this academic research project.

### 2.4 Retention
Data will be retained on the BUas GPU server for the duration of the project and for a reasonable period thereafter in accordance with BUas data retention policies, to allow verification of results if required.

---

## 3. Privacy and GDPR Considerations

**GDPR does not apply to this research.**

The dataset consists exclusively of greyscale morphometric images of *Arabidopsis thaliana* root systems — a plant species. No data relating to identified or identifiable natural persons is collected, processed, or stored at any point in this research. There are no human participants, no personal identifiers, no biometric data, and no sensitive information of any kind.

As defined under **GDPR Article 4(1)**, personal data means "any information relating to an identified or identifiable natural person." Since this research involves only plant root images, GDPR is not triggered.

**Justification summary:**
- No human subjects involved at any stage
- No personal data collected or processed
- No consent procedures required
- No data subject rights apply
- No Data Protection Impact Assessment (DPIA) required

---

## 4. FAIR Data Principles

### Findable
The dataset is clearly described in the research proposal (`Proposal_Asmamoghimi.pdf`) and research notebook (`Asmamoghimi_research_notebook.ipynb`), including its source (NPEC Hades system), cohort structure (Y2B_23, Y2B_24, Y2B_25), total size (492 image–mask pairs), and key characteristics (class imbalance, condensation variation). All experimental outputs are saved with consistent, descriptive naming conventions.

**Limitation:** The dataset is not publicly registered with a persistent identifier (e.g. DOI) as it is owned by BUas and provided for internal educational use only.

### Accessible
All data and code are accessible to the researcher and BUas supervisory staff via the BUas GPU server. The research notebook is fully self-contained and documents all steps required to reproduce the analysis from the stored data.

**Limitation:** The dataset is not publicly accessible, as it is proprietary to BUas/NPEC. However, the methodology is fully documented so that the study could be reproduced using a comparable plant root imaging dataset.

### Interoperable
- All images and masks are stored in **standard PNG format**, readable by any image processing library (Python, OpenCV, PIL, etc.)
- All code is written in **Python** using open-source libraries (PyTorch, NumPy, SciPy, Matplotlib), which are universally available
- The Jupyter notebook format (`.ipynb`) is an open, widely supported standard
- No proprietary file formats or tools are used at any stage

### Reusable
The research is fully documented to support reuse and reproduction:
- The research proposal describes the complete methodology, preprocessing pipeline, hyperparameters, and statistical analysis approach
- The EDA notebook (`EDA_Asmamoghimi.ipynb`) documents all exploratory findings and baseline results
- The research notebook (`Asmamoghimi_research_notebook.ipynb`) contains all implementation steps, results, and interpretation
- A fixed random seed (42) is used throughout, enabling exact reproduction of all results
- Deviations from the original proposal (e.g. error map update interval changed from 5 to 10 epochs due to GPU constraints) are explicitly documented in the notebook

**Limitation:** The raw dataset cannot be shared publicly as it is owned by BUas and derived from NPEC infrastructure. However, all code, preprocessing logic, and trained model checkpoints are available on the BUas server, and the methodology is described in sufficient detail to replicate the study on any comparable plant root imaging dataset.

---

## 5. Summary Table

| Aspect | Decision |
|--------|----------|
| Data source | NPEC Hades phenotyping system (provided by BUas) |
| Data type | Greyscale plant root images + binary masks |
| Human subjects | None |
| GDPR applicable | No — plant images only, no personal data |
| Storage location | BUas GPU server (/home/y2b/) only |
| Personal devices | Never used |
| Original data modified | Never — preprocessing produces copies only |
| Reproducibility | Full — fixed random seed 42 throughout |
| Public availability | Not applicable — proprietary BUas/NPEC dataset |
| FAIR compliance | Partially — fully interoperable and reusable; findability and accessibility limited by data ownership |

