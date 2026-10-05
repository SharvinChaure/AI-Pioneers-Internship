"""
Week 1 Task: Python for Machine Learning & Data Preprocessing
Cleans and preprocesses a sample customer dataset using NumPy and Pandas.
Run:  python data_preprocessing.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
log = {}  # collects results for the report


# ------------------------------------------------------------------
# STEP 0: Create a sample raw dataset (with deliberate data problems)
# ------------------------------------------------------------------
def make_raw_dataset(n=200):
    cities = ["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"]
    df = pd.DataFrame({
        "CustomerID": np.arange(1001, 1001 + n),
        "Age": rng.integers(18, 65, n).astype(float),
        "Gender": rng.choice(["Male", "Female"], n),
        "City": rng.choice(cities, n),
        "Education": rng.choice(["High School", "Graduate", "Postgraduate"], n,
                                p=[0.3, 0.5, 0.2]),
        "AnnualIncome": rng.normal(600000, 180000, n).round(0),
        "SpendingScore": rng.integers(1, 100, n).astype(float),
        "YearsWithCompany": rng.integers(0, 15, n).astype(float),
    })
    df["Purchased"] = ((df["SpendingScore"] + df["AnnualIncome"] / 10000
                        + rng.normal(0, 15, n)) > 100).map({True: "Yes", False: "No"})
    # constant / redundant column (to demonstrate feature selection)
    df["Country"] = "India"
    # redundant, highly correlated column
    df["IncomeMonthly"] = (df["AnnualIncome"] / 12).round(0)

    # inject missing values
    for col, k in [("Age", 14), ("AnnualIncome", 18), ("SpendingScore", 10),
                   ("City", 8), ("Education", 6)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    # inject outliers
    df.loc[[5, 60, 120], "AnnualIncome"] = [4500000, 5200000, 3900000]
    df.loc[[33], "Age"] = 150
    # inconsistent text formatting
    df.loc[[2, 9, 41], "City"] = ["  pune ", "MUMBAI", "delhi"]
    df.loc[[7, 15], "Gender"] = ["male", "FEMALE"]
    # duplicate rows
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)
    return df


df = make_raw_dataset()
df.to_csv("raw_customer_data.csv", index=False)

# ------------------------------------------------------------------
# STEP 1: Data loading & initial inspection
# ------------------------------------------------------------------
df = pd.read_csv("raw_customer_data.csv")
log["shape_raw"] = df.shape
log["dtypes"] = df.dtypes.astype(str).to_dict()
log["missing_raw"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
log["duplicates_raw"] = int(df.duplicated().sum())
log["head"] = df.head(5).fillna("NaN").astype(str).values.tolist()
log["head_cols"] = list(df.columns)

# Chart 1: missing values
miss = df.isnull().sum()
miss = miss[miss > 0].sort_values()
plt.figure(figsize=(6, 3.2))
plt.barh(miss.index, miss.values, color="#1F4E79")
plt.xlabel("Number of missing values")
plt.title("Missing Values per Column (Raw Data)")
plt.tight_layout()
plt.savefig("chart_missing.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# STEP 2: Data cleaning (duplicates + inconsistent text)
# ------------------------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)
log["shape_after_dup"] = df.shape

df["City"] = df["City"].str.strip().str.title()
df["Gender"] = df["Gender"].str.strip().str.title()
log["city_values"] = sorted(df["City"].dropna().unique().tolist())
log["gender_values"] = sorted(df["Gender"].dropna().unique().tolist())

# ------------------------------------------------------------------
# STEP 3: Handling outliers (IQR method) - done before imputation
#         so extreme values do not distort the mean/median
# ------------------------------------------------------------------
df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan  # impossible ages

Q1, Q3 = df["AnnualIncome"].quantile([0.25, 0.75])
IQR = Q3 - Q1
low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
out_mask = (df["AnnualIncome"] < low) | (df["AnnualIncome"] > high)
log["iqr"] = {"Q1": float(Q1), "Q3": float(Q3), "low": float(low),
              "high": float(high), "n_outliers": int(out_mask.sum())}

before_income = df["AnnualIncome"].copy()
df["AnnualIncome"] = df["AnnualIncome"].clip(lower=low, upper=high)  # capping

# Chart 2: boxplot before vs after
plt.figure(figsize=(6, 3.2))
plt.boxplot([before_income.dropna(), df["AnnualIncome"].dropna()],
            tick_labels=["Before capping", "After capping"], vert=False)
plt.title("Annual Income - Outlier Treatment (IQR)")
plt.tight_layout()
plt.savefig("chart_outliers.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# STEP 4: Handling missing values
# ------------------------------------------------------------------
log["missing_before_impute"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
num_fill = {}
for col in ["Age", "AnnualIncome", "SpendingScore"]:
    med = float(df[col].median())
    num_fill[col] = round(med, 2)
    df[col] = df[col].fillna(med)            # numeric -> median
cat_fill = {}
for col in ["City", "Education"]:
    mode = df[col].mode()[0]
    cat_fill[col] = mode
    df[col] = df[col].fillna(mode)           # categorical -> mode
log["num_fill"] = num_fill
log["cat_fill"] = cat_fill
log["missing_after"] = int(df.isnull().sum().sum())

# ------------------------------------------------------------------
# STEP 5: Feature selection
# ------------------------------------------------------------------
dropped = []
# (a) identifier - no predictive value
df = df.drop(columns=["CustomerID"]); dropped.append("CustomerID (unique identifier)")
# (b) constant column
const = [c for c in df.columns if df[c].nunique() == 1]
df = df.drop(columns=const); dropped += [f"{c} (constant value)" for c in const]
# (c) highly correlated numeric features
corr = df.select_dtypes("number").corr().abs()
log["corr_income"] = round(float(corr.loc["AnnualIncome", "IncomeMonthly"]), 3)
df = df.drop(columns=["IncomeMonthly"])
dropped.append("IncomeMonthly (redundant: AnnualIncome / 12, correlation %.2f)" % log["corr_income"])
log["dropped"] = dropped

corr2 = df.select_dtypes("number").corr()
plt.figure(figsize=(5, 4))
plt.imshow(corr2, cmap="Blues", vmin=-1, vmax=1)
plt.xticks(range(len(corr2)), corr2.columns, rotation=45, ha="right")
plt.yticks(range(len(corr2)), corr2.columns)
for i in range(len(corr2)):
    for j in range(len(corr2)):
        plt.text(j, i, f"{corr2.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("chart_corr.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# STEP 6: Encoding categorical variables
# ------------------------------------------------------------------
df["Purchased"] = df["Purchased"].map({"No": 0, "Yes": 1})             # binary target
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})               # binary label
df["Education"] = df["Education"].map({"High School": 0, "Graduate": 1,
                                       "Postgraduate": 2})              # ordinal
df = pd.get_dummies(df, columns=["City"], prefix="City", dtype=int)     # one-hot (nominal)
log["encoded_cols"] = list(df.columns)

# ------------------------------------------------------------------
# STEP 7: Normalization / scaling
# ------------------------------------------------------------------
scale_cols = ["Age", "AnnualIncome", "SpendingScore", "YearsWithCompany"]
log["stats_before_scale"] = df[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()

scaled_minmax = df.copy()
for c in scale_cols:  # Min-Max -> [0, 1]
    scaled_minmax[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

scaled_z = df.copy()
for c in scale_cols:  # Standardization (z-score)
    scaled_z[c] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

log["stats_minmax"] = scaled_minmax[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()
log["stats_z"] = scaled_z[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()

# ------------------------------------------------------------------
# STEP 8: Exploratory data analysis (EDA)
# ------------------------------------------------------------------
log["purchase_rate"] = round(float(df["Purchased"].mean()) * 100, 1)
log["corr_target"] = df[["Age", "AnnualIncome", "SpendingScore",
                         "YearsWithCompany", "Purchased"]].corr()["Purchased"].round(3).to_dict()
log["income_by_purchase"] = df.groupby("Purchased")["AnnualIncome"].mean().round(0).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].hist(df["Age"], bins=12, color="#1F4E79", edgecolor="white")
ax[0].set_title("Age Distribution"); ax[0].set_xlabel("Age")
df.groupby("Purchased")["SpendingScore"].mean().plot(kind="bar", ax=ax[1], color=["#9DB7D5", "#1F4E79"])
ax[1].set_title("Avg Spending Score by Purchase"); ax[1].set_xlabel("Purchased (0 = No, 1 = Yes)")
ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("chart_eda.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# STEP 9: Save final dataset
# ------------------------------------------------------------------
final = scaled_minmax
final.to_csv("cleaned_customer_data.csv", index=False)
log["shape_final"] = final.shape
log["final_head"] = final.head(5).round(3).astype(str).values.tolist()
log["final_cols"] = list(final.columns)

with open("log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)
print("Done. Final shape:", final.shape)
