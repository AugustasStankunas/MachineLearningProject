import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib import pyplot as plt
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

csv_path = "properties.csv"

df = pd.read_csv(csv_path, header=None, encoding="utf-8", na_values=["NA", "NaN", "", "Missing value", "missing value"])

df.columns = [
    "reference",
    "location",
    "price",
    "title",
    "bedrooms",
    "bathrooms",
    "indoor_sqm",
    "outdoor_sqm",
    "features"
]

def clean_price_to_eur(s):

    if pd.isna(s):
        return np.nan
    
    s = str(s)
    
    s = s.replace("â‚¬", "€")

    digits = "".join(ch for ch in s if ch.isdigit())
    
    if digits == "":
        return np.nan
    
    return float(digits)

df["price_eur"] = df["price"].apply(clean_price_to_eur)

df = df.dropna(subset=["price_eur"])

for col in ["bedrooms", "bathrooms", "indoor_sqm", "outdoor_sqm"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

numeric_cols = ["bedrooms", "bathrooms", "indoor_sqm", "outdoor_sqm"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def extract_property_type(title):
    if pd.isna(title):
        return np.nan

    t = str(title).strip()

    t = re.sub(r"\s+", " ", t)

    m = re.match(r"^\d+\s+Bedrooms?\s+(.*)$", t, flags=re.IGNORECASE)
    if m:
        prop_type = m.group(1).strip()
        return prop_type if prop_type else t

    m = re.match(r"^\d+\s+(.*)$", t)
    if m:
        prop_type = m.group(1).strip()
        return prop_type if prop_type else t

    return t

df["property_type"] = df["title"].apply(extract_property_type)

corr = ["price_eur"] + numeric_cols

corr_pearson = df[corr].corr(method="pearson")
corr_spearman = df[corr].corr(method="spearman")

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

for col in corr:
    low, high = iqr_bounds(df[col])
    outlier_mask = (df[col] < low) | (df[col] > high)
    n_out = outlier_mask.sum()

df_clean = df.copy()

df_clean = df_clean[df_clean["price_eur"] > 0]
df_clean = df_clean[(df_clean["bedrooms"] >= 0) & (df_clean["bedrooms"] <= 20)]
df_clean = df_clean[(df_clean["bathrooms"] >= 0) & (df_clean["bathrooms"] <= 20)]
df_clean = df_clean[(df_clean["indoor_sqm"] > 0) & (df_clean["indoor_sqm"] <= 2000)]
df_clean = df_clean[(df_clean["outdoor_sqm"] > 0) & (df_clean["outdoor_sqm"] <= 100000)]

cols_to_cap = ["price_eur", "indoor_sqm", "outdoor_sqm"]

caps = {}
for col in cols_to_cap:
    low_q = df_clean[col].quantile(0.01)
    high_q = df_clean[col].quantile(0.99)
    caps[col] = (low_q, high_q)

for col, (low_q, high_q) in caps.items():
    df_clean[col] = df_clean[col].clip(lower=low_q, upper=high_q)

def merge_rare_categories(df, col, min_freq=0.01):
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < min_freq].index
    other_label = f"Other_{col}"
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df

df_clean = merge_rare_categories(df_clean, "location", min_freq=0.01)
df_clean = merge_rare_categories(df_clean, "property_type", min_freq=0.01)

cat_cols = ["location", "property_type"]

df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

df_encoded["features_list"] = (
    df_clean["features"]
    .fillna("")
    .apply(lambda x: [f.strip() for f in str(x).split("|") if f.strip() != ""])
)

mlb = MultiLabelBinarizer()

features_matrix = mlb.fit_transform(df_encoded["features_list"])
feature_names = [f"feat_{f}" for f in mlb.classes_]

features_df = pd.DataFrame(
    features_matrix,
    columns=feature_names,
    index=df_encoded.index
)

df_model = pd.concat(
    [df_encoded.drop(columns=["features", "features_list"]) , features_df],
    axis=1
)

y = df_model["price_eur"]

X = df_model.drop(
    columns=[
        "price_eur",
        "price",
        "reference",
        "title"
    ],
    errors="ignore"
)

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

tree = DecisionTreeRegressor(
    max_depth=30,
    min_samples_leaf=50,
    random_state=0
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.0f} €")
print(f"R²: {r2:.3f}")