# Beyond Random Sampling: U-Net Patch Sampling Strategies for Plant Root Segmentation

Individual research project — BUas Block C, Year 2 ("Research Methodology and Statistics"). Tests whether *how* you sample training patches — not the model architecture — is the lever that fixes U-Net segmentation performance on noisy, class-imbalanced plant-root images.

> Solo research project: literature review, experimental design, U-Net implementation, statistical analysis, poster, and proposal are all my own work, completed within a research cohort at Breda University of Applied Sciences.

## Research Question

> Does patch sampling strategy affect U-Net segmentation performance on plant root images under condensation and noise?

**Background:** U-Net automates root segmentation from microscopy images, but standard random patch sampling has a structural flaw on this dataset — 85.5% of randomly sampled patches contain zero root pixels, because the mean root-pixel ratio across the dataset is only 0.71%. The model ends up training mostly on uninformative background.

**Hypotheses:**
- **H1** — Weighted sampling (patches drawn proportional to local foreground density) achieves higher Dice/IoU than random sampling.
- **H2** — Hard-example sampling (patches drawn proportional to the model's own error map, OHEM-inspired) achieves the highest Dice/IoU of all three strategies.

## Dataset

Greyscale morphometric scans of *Arabidopsis thaliana* root systems, collected by the Hades in-vitro phenotyping system at NPEC (Netherlands Plant Eco-phenotyping Centre). 492 image–mask pairs across three cohorts:

| Cohort | Images |
|---|---|
| Y2B_23 | 115 |
| Y2B_24 | 314 |
| Y2B_25 | 63 |

Split 70/15/15 (344 train / 74 val / 74 test), stratified by cohort, seed 42. Visual noise comes from condensation forming on the petri dish surface during imaging.

> Raw dataset images are owned by BUas/NPEC and are not redistributed in this repository — only my own code, written analysis, and results are included.

## Method

- **Architecture:** U-Net, 4 encoder blocks, channels 64/128/256/512, ~31M parameters.
- **Training:** Adam (lr=0.001) with `ReduceLROnPlateau`, batch size 8, combined loss = 0.8 × Dice + 0.2 × BCE, seed 42, trained on an NVIDIA RTX 6000 Ada Generation GPU.
- **Sampling strategies compared:**
  - *Random* — baseline, uniform patch sampling.
  - *Weighted* — patches sampled proportional to local foreground (root) density, computed via `scipy.ndimage.uniform_filter`.
  - *Hard-Example* — patches sampled proportional to the current model's error map (sliding-window inference), refreshed every 10 epochs.
- **Evaluation:** Dice coefficient (primary) and IoU (secondary) on a held-out test set; Wilcoxon signed-rank test (α=0.05), Cohen's d, and bootstrap 95% confidence intervals (1,000 iterations) for statistical comparison.

A few implementation details changed from the original proposal during execution (GPU assignment, the density-computation method, and how often error maps refresh) — all engineering adaptations to shared-server constraints, not changes to the experimental design itself.

## Results

| Strategy | Test Dice | Test IoU | Δ Dice vs. baseline |
|---|---|---|---|
| Random (baseline) | 0.6541 | 0.4282 | — |
| Weighted (H1) | 0.7241 | 0.4593 | +0.0701 |
| Hard-Example (H2) | 0.7670 | 0.5093 | +0.1129 |

Both hypotheses supported:

- **H1** (Weighted > Random): Wilcoxon p = 0.0001, Cohen's d = 0.21
- **H2** (Hard-Example > Random): Wilcoxon p < 0.0001, Cohen's d = 0.36 — also significantly beats Weighted (p < 0.0001, d = 0.15)

Hard-example sampling gives the largest and most statistically robust gain: a 17% relative improvement in Dice over random sampling, achieved with zero changes to the model architecture.

## Ethics & Data Management

Reviewed and approved by BUas supervisory staff under the BUas Research Ethics Review Board process — classified **low risk**: no human participants, no personal data, GDPR not triggered (the dataset is exclusively plant images). Funded entirely through the standard BUas curriculum, no external funding or conflicts of interest.

Data management followed a written plan: all data and model checkpoints stayed on the BUas GPU server (`/home/y2b/`) only, never on personal devices or cloud storage; the original raw dataset was never modified, only copied for preprocessing; access was restricted to me and BUas supervisory staff; a fixed seed (42) made every result reproducible end to end.

## Risk Management

Risks were tracked from project start through final submission (10 identified, logged with dated entries as they materialized). The most consequential:

| Risk | Response | Outcome |
|---|---|---|
| Severe class imbalance could make baseline training fail outright | Confirmed via EDA before training; designed H1/H2 explicitly to counter it | Resolved — both strategies improved on baseline |
| Single training run per strategy — can't separate strategy effect from random variation | Applied Wilcoxon + Cohen's d + bootstrap CIs instead of relying on raw score gaps; documented as a limitation | Accepted, statistically mitigated |
| Small test set (74 images) limits statistical power | Non-parametric testing (no normality assumption) + 1,000-iteration bootstrap CIs | Accepted, mitigated |
| GPU memory constraints forced hard-example error maps to refresh every 10 epochs instead of 5 | Documented as a deviation; monitored validation Dice for stagnation | Realized, managed — logged in notebook |
| Identical hyperparameters across all three strategies could favor one over another | Kept deliberately identical to isolate the sampling-strategy effect (correct controlled-experiment design); flagged as a limitation rather than tuned away | Accepted by design |

## Limitations

- Small test set (74 images) — limited statistical power.
- Only one training run per strategy — results may vary across runs.
- Single U-Net architecture — findings may not generalize to attention U-Net or nnU-Net.
- Single dataset domain — one lab, one plant species.
- Error maps updated every 10 epochs instead of 5 (GPU memory constraint), which may understate hard-example sampling's potential.

## Future Work

Repeat each experiment 3–5× to quantify run-to-run variability; test a hybrid weighted + hard-example strategy; evaluate on attention U-Net, nnU-Net, and TransUNet; validate on larger, multi-species datasets; move to fully online OHEM with per-batch error map updates; evaluate at full-image resolution rather than patch-level only.

## Repository Structure

```
.
├── README.md
├── research_proposal.pdf      # full proposal: literature review, methodology, stats plan
├── eda_notebook.ipynb          # exploratory data analysis
├── research_notebook.ipynb     # implementation, training, evaluation, hypothesis testing
├── poster.pdf                  # A0 research poster (final presentation)
├── risk_log.md                 # risk register with dated entries, tracked start to finish
├── data_management_plan.md     # storage, access control, FAIR compliance
└── ethics_application.docx     # BUas Research Ethics Review Board application (approved, low risk)
```

## Tech Stack

`PyTorch` · `NumPy` · `pandas` · `SciPy` (`scipy.ndimage` for density-based sampling) · `scikit-learn` · `Matplotlib` · `PIL`

## Skills Demonstrated

Literature review and hypothesis formulation, controlled experimental design (fixed seeds, stratified splits), U-Net implementation from scratch, custom data sampling pipelines, statistical hypothesis testing (Wilcoxon signed-rank, Cohen's d, bootstrap confidence intervals), scientific writing and poster design.

## References

Berger, J.-O., Castellan, G., Niaf, E., and Lartizien, C. (2017). iSample: A patch sampling algorithm for deep learning-based medical image segmentation. *ISBI*, pages 680–683.
Ciga, O., Xu, T., and Martel, A. L. (2021). Self supervised contrastive learning for digital histopathology. *Machine Learning with Applications*, 6:100198.
Fischer, P., Zimmerer, D., Küstner, T., and Maier-Hein, K. (2025). Progressive growing of patch size for medical image segmentation. *Medical Image Analysis*, 101:103420.
Han, X., Zheng, H., and Chen, M. (2023). Improved U-Net for plant root segmentation in minirhizotron images. *Frontiers in Plant Science*, 14:1143731.
Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, pages 234–241.
Schmidt-Mengin, M., Jobidon, V., Doré, F., Bhattacharya, P., Arbel, T., and Leppert, I. R. (2022). Online hard example mining versus fixed foreground oversampling for 3D U-Net segmentation of new MS lesions. *arXiv:2205.01626*.
Shrivastava, A., Gupta, A., and Girshick, R. (2016). Training region-based object detectors with online hard example mining. *CVPR*, pages 761–769.
Smith, A. G., Petersen, J., Selvan, R., and Rasmussen, C. R. (2020). Segmentation of roots in soil with U-Net. *Plant Methods*, 16(1):13.
Tappeiner, E., Welk, A., Hänsch, R., and Menze, B. (2022). Class-adaptive loss weighting and patch sampling for imbalanced medical image segmentation. *MICCAI 2022*, pages 311–320.
Teramoto, S. and Uga, Y. (2024). Deep learning-based segmentation of rice root systems from X-ray CT images under varying noise conditions. *Plant Methods*, 20(1):45.

## Acknowledgments

Conducted as an individual research deliverable for the "Research Methodology and Statistics" module at Breda University of Applied Sciences, within a research cohort. Dataset provided by NPEC; remains the property of BUas/NPEC and is not included in this repository.
