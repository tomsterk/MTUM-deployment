# MTUM Phase 2 Deployment

Production deployment of the Multi-Treatment Uplift Model (MTUM) for customer win-back targeting. This folder contains everything needed to (re)train the CatBoost model and score a deployment cohort to produce the customer-level incentive assignment file used by the CRM team.

---

## Uplift score calculation per incentive (MMOA)

For each customer–incentive pair, the CatBoost MultiClass model predicts four joint probabilities — one per (treatment, outcome) cell. Example for  customer with the €5 voucher arm:

| Cell                          | Probability |
|-------------------------------|-------------|
| €5 voucher, reactivated       | 5%          |
| €5 voucher, not reactivated   | 45%         |
| control, reactivated          | 3%          |
| control, not reactivated      | 47%         |

Under the Modified Outcome Approach (MMOA), the problem of uplift estimation is translated to estimating the probability of treatment/control and reactivated/not reactivated. Two outcomes are considered as positive, and two as negative:

- **Positive:** treatment + reactivated  and  control + not reactivated
- **Negative:** treatment + not reactivated  and  control + reactivated

The uplift score is Poitive − negative:

```
uplift_5eu    = (5% + 47%) − (45% + 3%)
                   =  52%       −  48%
                   = +4 pp
```

The same calculation runs for every incentive arm; the arm with the highest score is assigned to the customer. With unequal treatment priors (8 arms), each cell is reweighted by inverse propensity `1 / P(T=t)` before differencing — handled in `mtum_score.py` via the recomputed priors.

---

## Project Structure

```
Thesis code/
│
├── Queries Metabase/                                      # SQL queries used to generate customer datasets
│   ├── Pretreatment covariates.sql                        # Training covariates query
│   └── Pretreatment covariates deployement.sql            # Deployment covariates query
│
├── Data/
│   ├── covariates_modeling_uplift_models_2026-03-13.csv   # Training data (output of Pretreatment covariates.sql)
│   └── covariates_deployment_dataset_2026-03-17.csv       # Deployment data (output of Pretreatment covariates deployement.sql)
│
├── Models/
│   └── mtum_phase_2_catboost.cbm                          # Trained CatBoost model (produced by mtum_train.py)
│
├── Deployement_dataset/
│   └── treatment_selected_multi_2.csv                     # Customer-level incentive assignments (produced by mtum_score.py)
│
├── catboost_info/                                         # CatBoost training logs (auto-generated)
│
├── mtum_config.py                                         # Shared config and data-prep helpers
├── mtum_train.py                                          # Trains and saves the CatBoost model
├── mtum_score.py                                          # Scores deployment cohort, writes assignments
├── MTUM run script.ipynb                                  # Notebook to launch train + score from Jupyter
└── README.md
```

---

## Pipeline

```
Pretreatment covariates.sql              Pretreatment covariates deployement.sql
        │                                            │
        ▼                                            ▼
Data/covariates_modeling_*.csv          Data/covariates_deployment_*.csv
        │                                            │
        ▼                                            │
   mtum_train.py                                     │
        │                                            │
        ▼                                            │
Models/mtum_phase_2_catboost.cbm                     │
        │                                            │
        └────────────► mtum_score.py ◄───────────────┘
                            │
                            ▼
              Deployement_dataset/treatment_selected_multi_2.csv
```

---

## Scripts

### `mtum_config.py`
Shared module — not run directly. Holds:
- File paths for inputs (training/deployment CSVs) and outputs (model, assignment CSV)
- `TOP_PCT` (fraction of scored cohort to write out; `1.0` = everyone)
- Treatment encoding (which incentive arms to drop, integer mapping for `treatment_indicator`, mapping back to original incentive filenames)
- Feature schema (categorical columns, numeric columns, target column name)
- Data-prep helpers used by both training and scoring (`coerce_metrics_to_numeric`, `cast_dtypes`, `prepare_features`, `build_multi_outcome_target`, `get_treatment_probs_from_y_true`, etc.)

Edit this file to adjust paths, drop additional incentive arms, or change `TOP_PCT`.

### `mtum_train.py`
1. Loads training covariates and applies the prep pipeline (filter dropped arms, cast dtypes, build the `multi_outcome` target).
2. Trains a CatBoost MultiClass model with balanced class weights on the full training set.
3. Prints a classification report on training data as a sanity check.
4. Saves the trained model to `Models/mtum_phase_2_catboost.cbm`.

### `mtum_score.py`
1. Recomputes treatment priors `P(T=t)` from the training CSV (needed for the modified-outcome uplift estimator).
2. Loads the trained CatBoost model from disk.
3. Loads and preps the deployment cohort.
4. Scores each customer, computes per-arm uplift via the modified-outcome approach, and picks the optimal incentive per customer.
5. Optionally filters to the top X% by predicted uplift (controlled by `TOP_PCT` in `mtum_config.py`).
6. Writes the customer-level assignment CSV.
7. Prints summary stats: mean predicted CATE and an RFM (recency, monetary value) summary per assigned incentive.

---

## Running the pipeline

### From Jupyter
Open `MTUM run script.ipynb` and run the cell. It uses `sys.executable` so the scripts run with the same Python (and therefore the same packages) as the notebook kernel:

```python
import sys
!"{sys.executable}" mtum_train.py
!"{sys.executable}" mtum_score.py
```


Once the model is trained, only `mtum_score.py` needs to be re-run for a refreshed deployment cohort.

---

## Configuration

Common changes to `mtum_config.py`:

- **Drop an incentive arm:** add the arm's `_test_export.csv` and `_controle_export.csv` filenames to `DROP_INDICATORS`, and remove the corresponding integer key from `K_TO_FILENAME`.
- **Change targeting depth:** set `TOP_PCT` (e.g. `0.30` for top 30% by predicted uplift, `1.0` for everyone).
- **Update file paths:** edit `TRAINING_DATA_PATH`, `DEPLOYMENT_DATA_PATH`, `MODEL_OUTPUT_PATH`, `TREATMENT_OUTPUT_PATH`.
