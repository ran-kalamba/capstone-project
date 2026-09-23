# Predicting 30-Day Hospital Readmission Among Patients With Diabetes

This capstone project develops and evaluates machine-learning models that rank the risk of inpatient readmission within 30 days for patients with diabetes. It uses the public **Diabetes 130-US Hospitals for Years 1999–2008** dataset and compares logistic regression, random forest, and XGBoost models.

The final XGBoost model is intended as a research prototype for prioritizing beneficial post-discharge support. It is not a causal model or an autonomous clinical decision system.

## Project overview

- **Task:** Binary classification of readmission in less than 30 days
- **Unit of prediction:** One patient represented by the first eligible inpatient encounter
- **Prediction time:** Hospital discharge
- **Analytic cohort:** 69,973 patients
- **Positive cases:** 6,277 (8.97%)
- **Primary metric:** Average Precision, selected because the outcome is imbalanced
- **Models compared:** Dummy classifier, logistic regression, random forest, and XGBoost
- **Final model:** XGBoost

## Main results

On the held-out test set, the final XGBoost model achieved:

| Metric | Score |
| --- | ---: |
| Average Precision | 0.183 |
| ROC-AUC | 0.653 |
| Recall | 0.666 |
| Precision | 0.128 |
| Selected probability threshold | 0.0788 |

The operating threshold was selected from out-of-fold training predictions using Youden's J statistic. SHAP analysis identified discharge destination, previous inpatient utilization, length of stay, diagnosis group, laboratory activity, medication count, and age as important contributors to model predictions.

Fairness analysis found relatively small differences by gender, moderate differences by race, and substantial differences by age. Removing race, gender, and age did not consistently eliminate subgroup disparities.

## Repository contents

```text
.
├── notebooks/                                   # Complete analysis and modeling workflow in IPYNB file
├── Kavishna_Ranmali_capstone-final-report.pdf  # Written capstone report
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── data/                                       # Source data 
└── models/                                     # Model artefacts
└── figures/                                    # Figures generated in workflow
└── slidedecks/                                 # Technical and Business slide decks 
```

## Dataset

The project uses the [Diabetes 130-US Hospitals for Years 1999–2008 dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) from the UCI Machine Learning Repository.

The source dataset contains 101,766 inpatient encounters and 50 columns. Its inclusion criteria require a diabetic inpatient encounter, a hospital stay of 1–14 days, at least one laboratory test, and medication administration. The data therefore do not represent every person with diabetes or every hospital admission.

Reference: Strack et al. (2014), [Impact of HbA1c Measurement on Hospital Readmission Rates](https://doi.org/10.1155/2014/781670).

## Installation

Python 3.11 or later is recommended. The original notebook metadata records Python 3.13.5.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the analysis

Start JupyterLab:

```bash
jupyter lab
```

Open `diabetes_readmission_capstone.ipynb` and run the cells in order. The notebook performs:

1. Data loading and quality checks
2. Patient-level cohort construction and outcome definition
3. Domain-informed feature engineering
4. Stratified training and test split
5. Training-only exploratory data analysis
6. Preprocessing with imputation, scaling, and one-hot encoding
7. Baseline and tuned model evaluation
8. Logistic regression, random forest, and XGBoost comparison
9. Held-out test evaluation and threshold selection
10. SHAP and partial-dependence analysis
11. Fairness analysis by race, gender, and age
12. Sensitive-attribute removal analysis

Several tuning and cross-validation cells use parallel processing and may take time on machines with limited CPU or memory.

## Methodology notes

- The target maps `<30` to 1 and both `>30` and `NO` to 0.
- Only the first eligible encounter per patient is retained.
- Encounters ending in hospice or death are excluded.
- Patient and encounter identifiers are removed before modeling.
- Diagnosis codes are grouped into clinically interpretable ICD-9 categories.
- High-missingness, zero-variance, and very-low-exposure features are removed using documented rules.
- The data are split 80/20 with stratification and `random_state=42`.
- Exploratory decisions and transformation fitting use training data only.
- Five-fold stratified cross-validation is used for model development.
- The test set is reserved for final evaluation.

## Saving the final model

The final notebook cell can save the fitted XGBoost pipeline and its operating threshold to:

```text
models/final_xgboost_model_and_threshold.joblib
```

The artifact includes the fitted pipeline, selected threshold, primary metric, threshold-selection method, best parameters, and random seed. Run this cell only after all preceding training cells have completed successfully.

## Ethical use and limitations

This model should only be considered for prospective evaluation as a tool to offer additional post-discharge support, such as medication reconciliation, discharge counseling, early follow-up, or care-management calls.

It must not be used to:

- Deny or reduce care
- Determine insurance or treatment eligibility
- Make autonomous discharge decisions
- Assign blame to patients or clinicians
- Override clinical judgment or patient preferences

Important limitations include historical data from 1999–2008, possible changes in clinical practice and coding, incomplete capture of readmissions outside the source system, discarded longitudinal encounters, modest discrimination and precision, and subgroup performance differences. Any real-world use would require local prospective validation, human oversight, calibration review, fairness monitoring, and periodic performance reassessment.

## Reproducibility

Randomized operations use `random_state=42` where supported. The dependency file uses bounded version ranges so the project can receive compatible bug and security fixes. 

## Author

Kavishna Ranmali Kalamba Arachchi
