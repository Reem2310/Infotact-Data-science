# ============================================================
# WEEK 1: DATA ENGINEERING (STABILIZED VERSION)
# Manufacturing – Predictive Maintenance (CMAPSS)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ============================================================
# SETUP
# ============================================================

DATA_DIR = "CMaps"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILES = [
    "train_FD001.txt",
    "train_FD002.txt",
    "train_FD003.txt",
    "train_FD004.txt"
]

COLUMNS = (
    ['engine_id', 'cycle',
     'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def load_cmaps_file(file_path, columns):
    """
    Loads CMAPSS txt files. Handles space-delimited data.
    """
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python"
    )
    # Select only the columns we defined (ignoring trailing empty columns)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    return df

# ============================================================
# STEP 1: LOAD MULTIPLE DATA FILES
# ============================================================

df_list = []

if not os.path.exists(DATA_DIR):
    print(f"❌ Error: Folder '{DATA_DIR}' not found. Please create it and add .txt files.")
    sys.exit(1)

for file_name in DATA_FILES:
    full_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(full_path):
        print(f"⚠️ Warning: File not found: {full_path}. Skipping.")
        continue

    temp_df = load_cmaps_file(full_path, COLUMNS)
    temp_df["dataset_id"] = file_name.replace(".txt", "")
    df_list.append(temp_df)

if not df_list:
    print("❌ No data loaded. Check your file paths.")
    sys.exit(1)

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(["dataset_id", "engine_id", "cycle"]).reset_index(drop=True)

print(f"✅ Step 1: Loaded {len(df)} rows from {len(df_list)} files.")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

# Identify sensors with zero variance (no information)
sensor_cols = [c for c in df.columns if 'sensor' in c]
constant_cols = [c for c in sensor_cols if df[c].nunique() <= 1]

df.drop(columns=constant_cols, inplace=True)

# Correct way to Forward Fill within groups in modern Pandas
df = df.groupby(["dataset_id", "engine_id"], group_keys=False).apply(lambda x: x.ffill())

print(f"✅ Step 2: Cleaning completed. Dropped constant columns: {constant_cols}")

# ============================================================
# STEP 3: TEMPORAL FEATURE ENGINEERING
# ============================================================

# Ensure we only use sensors that weren't dropped in Step 2
LAG_FEATURES = [f for f in ["sensor_2", "sensor_3", "sensor_4"] if f in df.columns]

for lag in [1, 2]:
    for col in LAG_FEATURES:
        df[f"{col}_lag_{lag}"] = (
            df.groupby(["dataset_id", "engine_id"])[col]
            .shift(lag)
        )

WINDOW = 5
for col in LAG_FEATURES:
    df[f"{col}_roll_mean"] = (
        df.groupby(["dataset_id", "engine_id"])[col]
        .transform(lambda x: x.rolling(WINDOW).mean())
    )
    df[f"{col}_roll_std"] = (
        df.groupby(["dataset_id", "engine_id"])[col]
        .transform(lambda x: x.rolling(WINDOW).std())
    )

# Drop rows where lag/rolling calculations produced NaNs (the first 4 cycles of each engine)
df.dropna(inplace=True)

print("✅ Step 3: Lag & rolling features created.")

# ============================================================
# STEP 4: CREATE TARGET (RUL + FAILURE LABEL)
# ============================================================

# Calculate Remaining Useful Life (RUL)
max_cycle = (
    df.groupby(["dataset_id", "engine_id"])["cycle"]
    .max()
    .reset_index(name="max_cycle")
)

df = df.merge(max_cycle, on=["dataset_id", "engine_id"], how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]

# Binary classification: Will it fail within 24 cycles?
df["failure_24h"] = (df["RUL"] <= 24).astype(int)

df.drop(columns=["max_cycle"], inplace=True)

print("✅ Step 4: RUL & failure label created.")

# ============================================================
# STEP 5: CORRELATION MATRIX
# ============================================================

# Filter only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number]).drop(
    columns=["engine_id", "cycle", "RUL", "failure_24h"], errors='ignore'
)

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", linewidths=0.1)
plt.title("Sensor Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
plt.close()

print("✅ Step 5: Correlation matrix saved to 'output' folder.")

# ============================================================
# STEP 6: BASELINE MODEL
# ============================================================

# Drop non-feature columns
drop_cols = ["dataset_id", "engine_id", "cycle", "RUL", "failure_24h"]
X = df.drop(columns=drop_cols)
y = df["failure_24h"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sequential split (test on the end of the time series)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Step 6: Baseline Model Results")
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 7: SAVE OUTPUTS
# ============================================================

df.to_csv(f"{OUTPUT_DIR}/week1_feature_engineered_dataset.csv", index=False)

print("\n🎉 WEEK 1 DATA ENGINEERING COMPLETED SUCCESSFULLY")