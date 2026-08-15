# -*- coding: utf-8 -*-
# =========================================================
# SECTION 1: IMPORT DATA & LIBRARIES
# =========================================================

#Import some libraries
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

#Locate the dataset from the main project folder
project_root = Path(__file__).resolve().parents[1]
data_file = project_root / "data" / "ahs_price_sample.xlsx"
results_folder = project_root / "results"
results_folder.mkdir(exist_ok=True)

#Load the data
df = pd.read_excel(data_file)

#Create a row identifier
df.insert(0, "ID", range(1, len(df) + 1))
X = df.copy(deep=True)

#Data Exploration by Profiling Report
# profile = ProfileReport(X, explorative=True)
# profile.to_file("house_prices.html")

# =========================================================
# SECTION 2: HANDLE LEAKAGE
# =========================================================
# Remove exact duplicate rows
X = X.drop_duplicates(subset=X.columns.drop('ID')).copy()

# Keep target leakage column separately if it is needed to inspect later
X_target_leak = X[['ID', 'VALUE']].copy()

# Remove VALUE from predictors because target = LOGVALUE
X.drop(["VALUE"], axis=1, inplace=True)

# Separate target
y = X.pop("LOGVALUE")

# =========================================================
# SECTION 3: DATA CLEANING / PREPROCESSING
# =========================================================

# 3.1 Standardize blank strings to NaN, just in case
X = X.replace(r"^\s*$", np.nan, regex=True)

# 3.2 Convert AHS special survey codes to NaN
# These are your disguised missing / nonresponse codes
special_missing_codes = [-9, -8, -7, -6]
X = X.replace(special_missing_codes, np.nan)

# 3.3 Handle additional disguised missing values we identified
# Mean-imputation artifacts / suspicious values that should be NaN
disguised_missing_map = {
    "GARAGE": [1.140084050430258],
    "DISPL": [1.308708708708709],
    "FRSTOC": [1.490546927751519, 1.49054692775152],
    "WINTERNONE": [1.188785234226101],
    "UNITSF": [2300.971129476584],
    "LOT": [45103.41228452978]
}


for col, bad_values in disguised_missing_map.items():
    X[col] = X[col].replace(bad_values, np.nan)

# =========================================================
# SECTION 4: VARIABLE TYPE CONVERSION
# =========================================================
# Based on the dictionary + your interpretation

categorical_cols = [
    "REGION",
    "METRO",
    "CONDO",
    "CELLAR",
    "TUB",
    "GARAGE",
    "PORCH",
    "INCP",
    "EBAR",
    "MOBILTYP",
    "TYPE",
    "BUILT",
    "ELEV",
    "FRSTOC",
    "EVROD",
    "EROACH",
    "CRACKS",
    "HOLES",
    "WINTERNONE",
    "AIR",
    "AIRSYS",
    "DISPL",
    "DISH",
    "COOK"
]

numeric_cols = [
    "BATHS",
    "UNITSF",
    "LOT",
    "ROOMS",
    "DINING",
    "FLOORS",
    "NUNITS",
    "CLIMB"
]

# Convert categorical columns
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Convert numeric columns
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    

# =========================================================
# OPTIONAL FEATURE EXCLUSION: DROP EXTREMELY HIGH-MISSING COLUMNS
# =========================================================
missing_ratio = X.isna().mean()

# Practical rule for very sparse variables
high_missing_drop_cols = ['MOBILTYP', 'ELEV', 'INCP','FRSTOC']

X.drop(columns=high_missing_drop_cols, inplace=True)

# Update variable lists after dropping columns
categorical_cols = [col for col in categorical_cols if col in X.columns]
numeric_cols = [col for col in numeric_cols if col in X.columns]

print("\nDropped high-missing columns:")
print(high_missing_drop_cols)
print("X shape after dropping high-missing columns:", X.shape)


# =========================================================
# SECTION 5: QUICK CHECKS
# =========================================================
print("X shape after cleaning:", X.shape)
print("y shape:", y.shape)
print("\nMissing values by column:")
print(X.isna().sum().sort_values(ascending=False))

print("\nData types:")
print(X.dtypes)

# =========================================================
# SECTION 6: TRAIN / TEST SPLIT
# =========================================================
# Separate ID from predictors before modeling
row_id = X.pop('ID')

#Split training and test data
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, row_id, test_size=0.2, random_state=42
)


# =========================================================
# SECTION 7: EXPORT TRAIN / HOLDOUT FOR DATAROBOT
# =========================================================
# Train Dataset Export File
regression_train_set = X_train.copy(deep=True)
regression_train_set.insert(0, 'ID', id_train.values)
regression_train_set['LOGVALUE'] = y_train.values
regression_train_set.to_excel(results_folder / 'Regression_Train_Set.xlsx', header=True, index=False)

# Test Dataset Export File
regression_holdout_set = X_test.copy(deep=True)
regression_holdout_set.insert(0, 'ID', id_test.values)
regression_holdout_set['LOGVALUE'] = y_test.values
regression_holdout_set.to_excel(results_folder / 'Regression_Holdout_Set.xlsx', header=True, index=False)


#=========================================================
# SECTION 8: TPOT REGRESSION MODEL
#=========================================================
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Make copies
X_train_tpot_2 = X_train.copy(deep=True)
X_test_tpot_2 = X_test.copy(deep=True)

# 8.1 Preprocess data based on variable type
# Categorical -> most frequent imputation + one-hot encoding
categorical_pipeline_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',
                             min_frequency=0.0001,
                             sparse=False))
])

# Numeric -> mean imputation
numeric_pipeline_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Combine preprocessing
preprocessor_2 = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline_2, categorical_cols),
        ('num', numeric_pipeline_2, numeric_cols)
    ]
)

# Fit on training set only, transform both sets
X_train_tpot_2 = preprocessor_2.fit_transform(X_train_tpot_2)
X_test_tpot_2 = preprocessor_2.transform(X_test_tpot_2)

# 8.2 Instantiate TPOT regressor
reg_model_2 = TPOTRegressor(
    generations=5,
    population_size=100,
    verbosity=2,
    random_state=42,                                                        
    n_jobs=-1
)

# 8.3 Fit TPOT model on training data
reg_model_2.fit(X_train_tpot_2, y_train)

#we want to retrain on all training data using the fitted pipeline
model_2 = reg_model_2.fitted_pipeline_
model_2.fit(X_train_tpot_2, y_train)

# 8.4 Evaluate on the training data
regression_training_2 = X_train.copy(deep=True)
regression_training_2.insert(0, 'ID', id_train.values)
regression_training_2['pred_LOGVALUE'] = model_2.predict(X_train_tpot_2)
regression_training_2['LOGVALUE'] = y_train.values

# Export predictions (training)
regression_training_2.to_excel(r'TPOT_regression_test_TPOT2_imputation_train.xlsx', header=True, index=False)

# 8.5 Predict on holdout data
regression_test_2 = X_test.copy(deep=True)
regression_test_2.insert(0, 'ID', id_test.values)
regression_test_2['pred_LOGVALUE'] = model_2.predict(X_test_tpot_2)
regression_test_2['LOGVALUE'] = y_test.values

# 8.6 Export predictions
regression_test_2.to_excel(results_folder / 'TPOT_regression_test_TPOT2_imputation_test.xlsx', header=True, index=False)







