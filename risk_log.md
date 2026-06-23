# Risk Log — U-Net Patch Sampling Strategy for Plant Root Segmentation
**Researcher:** Asma Moghimi  
**Project:** Beyond Random Sampling: Improving U-Net Performance on Noisy Plant Root Images  
**Project Period:** April 16, 2025 – June 8, 2025  
**Dataset:** NPEC Hades in-vitro phenotyping system — Arabidopsis thaliana (492 images)

---

## Risk Assessment Matrix

| Risk ID | Description | Category | Probability | Impact | Score | Priority | Response Strategy | Owner | Monitoring Plan |
|---------|-------------|----------|-------------|--------|-------|----------|-------------------|-------|-----------------|
| R001 | Severe class imbalance (85.5% of random patches contain zero root pixels) renders baseline U-Net training ineffective | Data | High (3) | High (3) | 9 | Critical | Mitigation | Asma Moghimi | Confirmed via EDA before training; root pixel ratio tracked per cohort |
| R002 | Hard-example error maps updated every 10 epochs instead of 5 due to GPU memory constraints, reducing sampling responsiveness | Computational | Medium (2) | Medium (2) | 4 | Moderate | Contingency | Asma Moghimi | Monitor validation Dice curve; if H2 stagnates, consider reducing batch size to allow more frequent updates |
| R003 | Small test set (74 images) limits statistical power and reliability of hypothesis testing | Data | High (3) | Medium (2) | 6 | Critical | Acceptance | Asma Moghimi | Apply Wilcoxon signed-rank test and report Cohen's d effect size; use 1,000-iteration bootstrap CIs to compensate |
| R004 | Only one training run per strategy — results may vary across runs due to stochastic training | Method | High (3) | Medium (2) | 6 | Critical | Acceptance | Asma Moghimi | Document single-run limitation explicitly; recommend 3–5 repeated runs in future work section |
| R005 | Hard-example sampling (H2) introduces computational overhead from per-image error map maintenance, potentially slowing training | Computational | Medium (2) | Medium (2) | 4 | Moderate | Mitigation | Asma Moghimi | Set error map update interval to every 10 epochs; monitor epoch training time; compare wall-clock time across strategies |
| R006 | Baseline hyperparameters may not be equally optimal for weighted and hard-example strategies, introducing confound | Method | Medium (2) | Medium (2) | 4 | Moderate | Acceptance | Asma Moghimi | Keep all hyperparameters identical across conditions by design; acknowledge as limitation in proposal |
| R007 | Dataset from single lab and single species (Arabidopsis thaliana) limits generalisability of findings | Data | High (3) | Low (1) | 3 | Moderate | Acceptance | Asma Moghimi | Explicitly state domain limitation in proposal and poster; propose multi-species validation as future work |
| R008 | Mild overfitting observed in baseline (val Dice 0.7892 vs test Dice 0.7285) may persist or worsen in proposed strategies | Method | Medium (2) | Medium (2) | 4 | Moderate | Mitigation | Asma Moghimi | Dropout at rate 0.2 applied after each pooling layer; early stopping with patience=10 monitored on validation Dice |
| R009 | Condensation variation across cohorts (Y2B_23, Y2B_24, Y2B_25) may cause test performance to be cohort-dependent rather than strategy-dependent | Data | Medium (2) | Medium (2) | 4 | Moderate | Mitigation | Asma Moghimi | Stratified split by cohort ensures proportional representation across train/val/test; cohort breakdown reported in dataset table |
| R010 | Results may not generalise beyond standard U-Net to architectures such as attention U-Net, nnU-Net, or TransUNet | Method | High (3) | Low (1) | 3 | Moderate | Acceptance | Asma Moghimi | Scope explicitly limited to standard U-Net; alternative architectures proposed as future work |

---

## Dated Entries

### Entry 1 — April 16, 2025 (Project Start / EDA Phase)

**Risk Identified:** R001 — Class imbalance severity  
**Description:** Initial EDA on the merged 492-image dataset revealed that the mean root pixel ratio is only 0.71% (median 0.17%). A patch simulation confirmed that under random 256×256 patch sampling, 85.5% of patches contain zero root pixels. Twenty images (4.1%) contain no root pixels at all.  
**Impact:** If training proceeds with random sampling as the only strategy, the model will receive gradient signal from root pixels in fewer than 15% of patches, making it highly likely to predict background everywhere. This directly threatens the validity of the baseline and the informativeness of the proposed strategies.  
**Mitigation Action Taken:** EDA completed and findings documented before any training began. Class imbalance metrics (root pixel ratio distribution, patch difficulty breakdown) integrated into research proposal Section 3.3. Weighted and hard-example sampling strategies designed explicitly to address this risk.  
**Trello Adjustment:** Task "Run EDA and document class imbalance" moved to Done. New task "Implement weighted sampling foreground probability map" added to Sprint 1 backlog with high priority.

---

### Entry 2 — April 23, 2025 (Baseline Training Complete)

**Risk Identified:** R008 — Baseline overfitting  
**Description:** Baseline U-Net trained with random sampling achieved a validation Dice of 0.7892 but a test Dice of 0.7285 — a gap of 0.064. Test IoU was notably low at 0.4005, indicating a high rate of false positive background predictions.  
**Impact:** The gap between validation and test performance suggests mild overfitting. The low IoU relative to Dice confirms the model is biased toward background prediction, consistent with training predominantly on empty patches. If the proposed strategies do not address this, the same pattern may repeat.  
**Mitigation Action Taken:** Dropout rate of 0.2 retained across all strategies. Early stopping with patience=10 monitoring validation Dice confirmed active. Baseline results documented as the official performance floor for hypothesis testing.  
**Trello Adjustment:** Task "Train baseline and record test metrics" moved to Done. Task "Confirm early stopping and dropout settings before H1/H2 runs" added to Sprint 2.

---

### Entry 3 — April 30, 2025 (Weighted Sampling Training — H1)

**Risk Identified:** R009 — Cohort-dependent performance variation  
**Description:** During H1 training, inspection of per-image Dice scores on the validation set suggested that images from cohort Y2B_25 (smallest cohort, 63 images) showed higher variance than Y2B_24. This raised the concern that performance differences between strategies might reflect cohort composition of the test set rather than the effect of sampling strategy.  
**Impact:** If the test set happened to contain a disproportionate number of low-quality Y2B_25 images, any strategy comparison would be confounded by cohort effects rather than attributable to sampling alone.  
**Mitigation Action Taken:** Verified that stratified splitting by cohort was correctly applied and that Y2B_25 is proportionally represented across train/val/test splits. Cohort breakdown confirmed in Table 1 of the proposal (Y2B_23: 17 test, Y2B_24: 47 test, Y2B_25: 9 test). No adjustment to the split was made as stratification was already in place.  
**Trello Adjustment:** Task "Verify cohort stratification in data split" added and completed within the sprint.

---

### Entry 4 — May 7, 2025 (Hard-Example Sampling Setup — H2)

**Risk Identified:** R002 — GPU constraints limiting error map update frequency  
**Description:** The original plan (per the proposal) intended to update per-image error maps every 5 epochs. During H2 setup, it became clear that computing and storing error maps for all 344 training images at this frequency, while also running training at batch size 8, caused GPU memory pressure on the NVIDIA RTX A6000. Update frequency was reduced to every 10 epochs to stabilise training.  
**Impact:** Less frequent error map updates mean the hard-example sampler responds more slowly to changes in model prediction errors. Condensation-affected patches that become "easy" for the model may continue to be oversampled for up to 10 epochs, and newly difficult patches may take longer to enter the sampling distribution. This may dampen H2's advantage over H1.  
**Mitigation Action Taken:** Update interval set to every 10 epochs and documented as a deviation from original plan in the research notebook (Section 1, "Deviations from Proposal"). Error map update epochs logged during training to verify updates were occurring. The limitation is acknowledged explicitly on the poster.  
**Trello Adjustment:** Task "Implement H2 error map with 5-epoch update" revised to "Implement H2 error map with 10-epoch update — GPU constraint" and moved to In Progress.

---

### Entry 5 — May 14, 2025 (H2 Training Complete — Results Review)

**Risk Identified:** R004 — Single training run per strategy  
**Description:** With H2 training complete, results showed Hard-Example sampling achieved Dice 0.7670, Weighted achieved 0.7241, and Baseline 0.6541. However, with only one training run per strategy, it is impossible to determine whether these differences reflect the strategies themselves or random variation from weight initialisation, patch ordering, and stochastic dropout.  
**Impact:** Statistical testing (Wilcoxon, p<0.001) confirms the results are significant across the 74 test images, but run-to-run variance cannot be quantified. A single favourable run for H2 might overstate its advantage.  
**Mitigation Action Taken:** Wilcoxon signed-rank test with Cohen's d and 95% bootstrap confidence intervals applied to all pairwise comparisons. Results reported with effect sizes (d=0.21 for H1 vs baseline; d=0.36 for H2 vs baseline). Single-run limitation documented in limitations section of poster and research notebook Section 4. Repeating each strategy 3–5 times added as the first future work recommendation.  
**Trello Adjustment:** Task "Run statistical tests on test-set Dice scores" moved to In Progress. Task "Document single-run limitation in notebook Section 4" added.

---

### Entry 6 — May 21, 2025 (Statistical Analysis and Hypothesis Testing)

**Risk Identified:** R003 — Small test set limiting statistical power  
**Description:** With 74 test images, the Wilcoxon signed-rank test produces statistically significant results (p=0.0001 for H1; p<0.001 for H2). However, effect sizes are small to moderate (d=0.21 and d=0.36), and the confidence intervals for the H2 vs H1 comparison are wider than desired, reflecting limited sample size.  
**Impact:** Small test set means that even statistically significant differences may not be practically robust. The observed advantage of H2 over H1 (Dice 0.7670 vs 0.7241) is meaningful but could shift with a larger or differently composed test set.  
**Mitigation Action Taken:** Bootstrap resampling with 1,000 iterations used to generate 95% CIs for mean Dice differences. Non-parametric test (Wilcoxon) chosen specifically because normality cannot be assumed with 74 samples. Statistical power limitation acknowledged as the first limitation on the poster.  
**Trello Adjustment:** Task "Apply Wilcoxon test and bootstrap CIs" moved to Done. Task "Write hypothesis test summary for notebook Section 3" added.

---

### Entry 7 — May 28, 2025 (Notebook Finalisation)

**Risk Identified:** R010 — Limited generalisability beyond standard U-Net  
**Description:** All experiments used a single standard U-Net architecture. Results may not transfer to attention U-Net, nnU-Net, or TransUNet, which use different feature aggregation and attention mechanisms that may interact differently with patch sampling strategies.  
**Impact:** Conclusions are scoped to standard U-Net on one dataset. The field may be more interested in whether the findings hold for state-of-the-art architectures, which limits the direct applicability of the work.  
**Mitigation Action Taken:** Architecture scope explicitly stated in proposal Section 6.3 and poster limitations. Evaluation on attention U-Net, nnU-Net, and TransUNet listed as future work items 3 and 6 on the poster.  
**Trello Adjustment:** Task "Finalise limitations and future work in notebook Section 4" added and completed.

---

### Entry 8 — June 4, 2025 (Final Review Before Submission)

**Risk Identified:** R006 — No per-strategy hyperparameter tuning  
**Description:** All three strategies used identical hyperparameters (lr=0.001, batch size=8, patience=10, dropout=0.2). No grid search or learning rate tuning was performed per strategy, which means the hyperparameter configuration optimal for random sampling may not be optimal for weighted or hard-example sampling.  
**Impact:** It is possible that H1 or H2 would perform better with a lower learning rate or higher dropout, which would understate their true advantage over the baseline. Alternatively, the current configuration may favour one strategy over another.  
**Mitigation Action Taken:** Identical hyperparameters across all conditions maintained by design to isolate the effect of sampling strategy — this is the correct controlled experimental approach. Acknowledged as a limitation in the proposal (Section 6.3) and in the research notebook. Per-strategy hyperparameter search proposed as future work.  
**Trello Adjustment:** No planning change required; limitation documented in final notebook version.

---

## Summary Table

| Risk ID | Status at Close | Outcome |
|---------|----------------|---------|
| R001 | Resolved | EDA confirmed imbalance; both H1 and H2 directly addressed it |
| R002 | Realised / Managed | Error maps updated every 10 epochs; documented as deviation |
| R003 | Accepted / Mitigated | Non-parametric tests and bootstrap CIs applied throughout |
| R004 | Accepted | Single runs completed; limitation documented; future work defined |
| R005 | Managed | Computational overhead monitored; update interval adjusted |
| R006 | Accepted | Identical hyperparameters maintained by design; acknowledged as limitation |
| R007 | Accepted | Single-domain dataset; generalisation limitation explicitly stated |
| R008 | Mitigated | Dropout and early stopping in place; val/test gap remained small |
| R009 | Mitigated | Stratified split verified; cohort breakdown confirmed in results |
| R010 | Accepted | Single architecture by design; multi-architecture testing as future work |
