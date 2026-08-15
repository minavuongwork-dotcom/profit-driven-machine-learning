
# -*- coding: utf-8 -*-
# =========================================================
# SECTION 1: IMPORT DATA & LIBRARIES
# =========================================================

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Locate the project folders
project_root = Path(__file__).resolve().parents[1]
data_file = project_root / "data" / "ahs_insurance_sample.xlsx"
results_folder = project_root / "results"
results_folder.mkdir(exist_ok=True)

# Load the data
df = pd.read_excel(data_file)

# Create a row identifier
df.insert(0, "ID", range(1, len(df) + 1))

# Work on a copy
X = df.copy(deep=True)

# Optional data profiling
# from ydata_profiling import ProfileReport
# profile = ProfileReport(X, explorative=True)
# profile.to_file(results_folder / "insurance_profile.html")

# # Optional profiling
# profile = ProfileReport(X, explorative=True)
# profile.to_file("insurance_profile_updated.html")


# =========================================================
# SECTION 2: TARGET + LEAKAGE
# =========================================================

# Remove exact duplicates except ID
X = X.drop_duplicates(subset=X.columns.drop("ID")).copy()

# Keep AMTI separately for later profit calculation
X_target_leak = X[["ID", "AMTI"]].copy()

# Separate target
y = X.pop("BUYI")

# Remove leakage variable from predictors
X.drop(["AMTI"], axis=1, inplace=True)

print("Original class balance:", Counter(y))


# =========================================================
# SECTION 3: DATA CLEANING / PREPROCESSING
# =========================================================

# 3.1 Standardize blank strings to NaN
X = X.replace(r"^\s*$", np.nan, regex=True)

# 3.2 Convert survey missing codes to NaN
special_missing_codes = [-9, -8, -7, -6]
X = X.replace(special_missing_codes, np.nan)

# 3.3 Additional disguised missing / invalid values
disguised_missing_map = {
    "CONFEE": [1110],
    "ZINCN": [-2049]
}

for col, bad_values in disguised_missing_map.items():
    if col in X.columns:
        X[col] = X[col].replace(bad_values, np.nan)

# 3.4 Create additional features
income = pd.to_numeric(X["ZINC2"], errors="coerce").replace(0, np.nan)
housing_cost = pd.to_numeric(X["ZSMHC"], errors="coerce")
housing_value = pd.to_numeric(X["VALUE"], errors="coerce")
built_year = pd.to_numeric(X["BUILT"], errors="coerce")

X["Housing_cost_to_income"] = housing_cost / income
X["Housing_value_to_income"] = housing_value / income

# The AHS data used in this project refer to 2011
X["home_age_group"] = 2011 - built_year

problem_cols = ["EVROD", "EROACH", "CRACKS", "HOLES"]
problem_data = X[problem_cols].apply(pd.to_numeric, errors="coerce")
X["problem_count"] = problem_data.eq(1).sum(axis=1)

# 3.5 Remove redundant income variable if present
if "ZINC" in X.columns:
    X.drop(["ZINC"], axis=1, inplace=True)
    
# 3.6 Define variable groups first
categorical_cols = [
    "CONFEE",
    "IFFEE",
    "HHSEX",
    "ZINCH",
    "QSS",
    "QSELF",
    "QRENT",
    "QRETIR",
    "REGION",
    "METRO3",
    "CONDO",
    "CELLAR",
    "MOBILTYP",
    "TYPE",
    "BUILT",
    "FRSTOC",
    "EVROD",
    "EROACH",
    "CRACKS",
    "HOLES",
    "WINTERNONE",
    "AIR",
    "AIRSYS"
]

numeric_cols = [
    "ZSMHC",
    "HHAGE",
    "ZINC2",
    "ZINCN",
    "VALUE",
    "UNITSF",
    "LOT",
    "ROOMS",
    "CLIMB",
    "Housing_cost_to_income",
    "Housing_value_to_income",
    "home_age_group",
    "problem_count"
]

# 3.7 Convert types
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype("category")

for col in numeric_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        
# =========================================================
# OPTIONAL FEATURE EXCLUSION: DROP EXTREMELY HIGH-MISSING COLUMNS
# =========================================================
missing_ratio = X.isna().mean()

# Practical rule for very sparse variables
high_missing_drop_cols = ["MOBILTYP", "CLIMB", "ZINCN", "CONFEE","FRSTOC"]

X.drop(columns=high_missing_drop_cols, inplace=True)

# Update variable lists after dropping columns
categorical_cols = [col for col in categorical_cols if col in X.columns]
numeric_cols = [col for col in numeric_cols if col in X.columns]

print("\nDropped high-missing columns:")
print(high_missing_drop_cols)
print("X shape after dropping high-missing columns:", X.shape)


# =========================================================
# SECTION 4: QUICK CHECKS
# =========================================================

print("\nX shape after cleaning:", X.shape)
print("y shape:", y.shape)

print("\nMissing values by column after dropping sparse columns:")
print(X.isna().sum().sort_values(ascending=False))

print("\nData types:")
print(X.dtypes)

print("\nTarget balance:")
print(y.value_counts(dropna=False))

# =========================================================
# SECTION 5: TRAIN / VALUE / HOLDOUT SPLIT
# =========================================================

row_id = X.pop("ID")

X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
    X, y, row_id,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_value, X_test, y_value, y_test, id_value, id_test = train_test_split(
    X_temp, y_temp, id_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTrain balance:", Counter(y_train))
print("Value balance:", Counter(y_value))
print("Holdout balance:", Counter(y_test))


# =========================================================
# SECTION 6: HELPER FUNCTION FOR TRAIN-ONLY IMPUTATION
# =========================================================

def prepare_for_resampling(train_df, other_df_list, categorical_cols, numeric_cols):
    """
    Prepare data for downsampling / SMOTENC:
    - fit imputers on training data only
    - categorical: most_frequent
    - numeric: median
    - align categorical levels across train / value / holdout
    - convert categoricals to integer codes for SMOTENC
    """

    X_train_ready = train_df.copy(deep=True)
    other_ready_list = [df.copy(deep=True) for df in other_df_list]

    cat_cols_present = [c for c in categorical_cols if c in X_train_ready.columns]
    num_cols_present = [c for c in numeric_cols if c in X_train_ready.columns]

    # Impute categorical columns
    if len(cat_cols_present) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_ready[cat_cols_present] = cat_imputer.fit_transform(X_train_ready[cat_cols_present])

        for i in range(len(other_ready_list)):
            other_ready_list[i][cat_cols_present] = cat_imputer.transform(
                other_ready_list[i][cat_cols_present]
            )

    # Impute numeric columns
    if len(num_cols_present) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X_train_ready[num_cols_present] = num_imputer.fit_transform(X_train_ready[num_cols_present])

        for i in range(len(other_ready_list)):
            other_ready_list[i][num_cols_present] = num_imputer.transform(
                other_ready_list[i][num_cols_present]
            )

    # Recast categoricals after imputation
    for col in cat_cols_present:
        X_train_ready[col] = X_train_ready[col].astype("category")
        for i in range(len(other_ready_list)):
            other_ready_list[i][col] = other_ready_list[i][col].astype("category")

    # Align category levels across train/value/holdout, then code as integers
    for col in cat_cols_present:
        combined_series = [X_train_ready[col]] + [df[col] for df in other_ready_list]
        combined_cats = pd.Categorical(pd.concat(combined_series, axis=0))
        categories = combined_cats.categories

        X_train_ready[col] = pd.Categorical(
            X_train_ready[col],
            categories=categories
        ).codes

        for i in range(len(other_ready_list)):
            other_ready_list[i][col] = pd.Categorical(
                other_ready_list[i][col],
                categories=categories
            ).codes

    # Make sure numeric columns stay numeric
    for col in num_cols_present:
        X_train_ready[col] = pd.to_numeric(X_train_ready[col], errors="coerce")
        for i in range(len(other_ready_list)):
            other_ready_list[i][col] = pd.to_numeric(
                other_ready_list[i][col], errors="coerce"
            )

    # Safety check
    print("\nRemaining missing values in resampling-ready training set:",
          X_train_ready.isna().sum().sum())

    categorical_feature_indices = [
        X_train_ready.columns.get_loc(col) for col in cat_cols_present
    ]

    return X_train_ready, other_ready_list, categorical_feature_indices


# Train-only imputation for resampled branches
X_train_ready, [X_value_ready, X_test_ready], categorical_feature_indices = prepare_for_resampling(
    X_train, [X_value, X_test], categorical_cols, numeric_cols
)

print("\nPrepared training shape:", X_train_ready.shape)
print("Prepared valuation shape:", X_value_ready.shape)
print("Prepared holdout shape:", X_test_ready.shape)


# =========================================================
# SECTION 7: RESAMPLING
# =========================================================

# Use the same prepared training set for both resampling branches
X_train_resample_base = X_train_ready.copy(deep=True)
y_train_resample_base = y_train.copy(deep=True)

# 8.1 Downsampling
rus = RandomUnderSampler(random_state=42)
X_downsampled, y_downsampled = rus.fit_resample(X_train_resample_base, y_train_resample_base)
print("\nDownsampled target shape:", Counter(y_downsampled))

# 8.2 SMOTENC
smotenc = SMOTENC(
    categorical_features=categorical_feature_indices,
    random_state=42,
    k_neighbors=5
)
X_smotenc, y_smotenc = smotenc.fit_resample(X_train_resample_base, y_train_resample_base)
print("SMOTENC target shape:", Counter(y_smotenc))


# =========================================================
# SECTION 8: EXPORT DATASETS FOR DATAROBOT
# =========================================================

# Downsampled training file for DataRobot
insurance_DR_downsampled = pd.DataFrame(X_downsampled, columns=X_train_ready.columns)
insurance_DR_downsampled.insert(0, "ID_train_resampled", range(1, len(insurance_DR_downsampled) + 1))
insurance_DR_downsampled["BUYI"] = y_downsampled.values
insurance_DR_downsampled.to_excel(results_folder / "insurance_DR_downsampled_training.xlsx", header=True, index=False)

# SMOTENC training file for DataRobot
insurance_DR_smotenc = pd.DataFrame(X_smotenc, columns=X_train_ready.columns)
insurance_DR_smotenc.insert(0, "ID_train_resampled", range(1, len(insurance_DR_smotenc) + 1))
insurance_DR_smotenc["BUYI"] = y_smotenc.values
insurance_DR_smotenc.to_excel(results_folder / "insurance_DR_smotenc_training.xlsx", header=True, index=False)

# Untouched valuation file for DataRobot / later profit evaluation
insurance_value = X_value.copy(deep=True)
insurance_value.insert(0, "ID", id_value.values)
insurance_value["BUYI"] = y_value.values

# merge AMTI back by ID
insurance_value = pd.merge(
    X_target_leak,
    insurance_value,
    on="ID",
    how="inner"
)

insurance_value.to_excel(results_folder / "insurance_value.xlsx", header=True, index=False)

# Untouched holdout file for DataRobot / later profit evaluation
insurance_holdout = X_test.copy(deep=True)
insurance_holdout.insert(0, "ID", id_test.values)
insurance_holdout["BUYI"] = y_test.values

# merge AMTI back by ID
insurance_holdout = pd.merge(
    X_target_leak,
    insurance_holdout,
    on="ID",
    how="inner"
)

insurance_holdout.to_excel(results_folder / "insurance_holdout.xlsx", header=True, index=False)


# =========================================================
# SECTION 9: ONE TPOT CLASSIFIER ONLY
# =========================================================

base_clf = TPOTClassifier(
    generations=5,
    population_size=100,
    verbosity=2,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# SECTION 10: HELPER FUNCTION FOR TPOT TRAINING + VALUATION
# =========================================================

def fit_value_tpot(
    base_clf,
    X_train_model,
    y_train_model,
    X_refit,
    y_refit,
    X_value_eval,
    y_value_eval,
    X_value_raw,
    id_value,
    X_target_leak,
    model_name
):
    """
    Train TPOT on one candidate resampled training set.
    Refit chosen pipeline on the same resampled training set.
    Predict on valuation set only.
    """

    print(f"\n{'='*70}")
    print(f"RUNNING: {model_name}")
    print(f"{'='*70}")

    clf = clone(base_clf, safe=True)
    clf.fit(X_train_model, y_train_model)

    # Refit chosen pipeline on same resampled data
    model = clf.fitted_pipeline_
    model.fit(X_refit, y_refit)

    # Predict on valuation only
    value_pred = model.predict(X_value_eval)
    value_prob = model.predict_proba(X_value_eval)[:, 1]

    value_output = X_value_raw.copy(deep=True)
    value_output.insert(0, "ID", id_value.values)
    value_output["pred_BUYI"] = value_pred
    value_output["pred_prob_BUYI"] = value_prob
    value_output["BUYI"] = y_value_eval.values

    # Merge back AMTI for later profit calculations
    value_output = pd.merge(
        X_target_leak,
        value_output,
        on="ID",
        how="inner"
    )

    value_output.to_excel(results_folder / f"{model_name}_value.xlsx", header=True, index=False)

    print(f"\n{model_name} - valuation confusion matrix")
    print(confusion_matrix(y_value_eval, value_pred))

    clf.export(results_folder / f"{model_name}_best_pipeline.py")

    return {
        "name": model_name,
        "tpot_object": clf,
        "model": model,
        "value_pred": value_pred,
        "value_prob": value_prob
    }


# =========================================================
# SECTION 11: TPOT RUNS FOR MODEL SELECTION
# Focus only on the two resampled branches
# =========================================================

# 11.1 TPOT - downsampled
results_downsampled = fit_value_tpot(
    base_clf=base_clf,
    X_train_model=X_downsampled,
    y_train_model=y_downsampled,
    X_refit=X_downsampled,
    y_refit=y_downsampled,
    X_value_eval=X_value_ready,
    y_value_eval=y_value,
    X_value_raw=X_value,
    id_value=id_value,
    X_target_leak=X_target_leak,
    model_name="TPOT_downsampled"
)

# 11.2 TPOT - SMOTENC
results_smotenc = fit_value_tpot(
    base_clf=base_clf,
    X_train_model=X_smotenc,
    y_train_model=y_smotenc,
    X_refit=X_smotenc,
    y_refit=y_smotenc,
    X_value_eval=X_value_ready,
    y_value_eval=y_value,
    X_value_raw=X_value,
    id_value=id_value,
    X_target_leak=X_target_leak,
    model_name="TPOT_smotenc"
)


# =========================================================
# SECTION 12: TPOT RUNS FOR MODEL SELECTION
# Focus only on the two resampled branches
# =========================================================

test_pred_smotenc = results_smotenc["model"].predict(X_test_ready)
test_prob_smotenc = results_smotenc["model"].predict_proba(X_test_ready)[:, 1]

holdout_smotenc = X_test.copy(deep=True)
holdout_smotenc.insert(0, "ID", id_test.values)
holdout_smotenc["pred_BUYI"] = test_pred_smotenc
holdout_smotenc["pred_prob_BUYI"] = test_prob_smotenc
holdout_smotenc["BUYI"] = y_test.values

holdout_smotenc = pd.merge(X_target_leak, holdout_smotenc, on="ID", how="inner")
holdout_smotenc.to_excel(results_folder / "TPOT_smotenc_holdout.xlsx", index=False)



test_pred_down = results_downsampled["model"].predict(X_test_ready)
test_prob_down = results_downsampled["model"].predict_proba(X_test_ready)[:, 1]

holdout_down = X_test.copy(deep=True)
holdout_down.insert(0, "ID", id_test.values)
holdout_down["pred_BUYI"] = test_pred_down
holdout_down["pred_prob_BUYI"] = test_prob_down
holdout_down["BUYI"] = y_test.values

holdout_down = pd.merge(X_target_leak, holdout_down, on="ID", how="inner")
holdout_down.to_excel(results_folder / "TPOT_downsampled_holdout.xlsx", index=False)