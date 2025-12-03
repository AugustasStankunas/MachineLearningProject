import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib import pyplot as plt
import re
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from image_processing import embed_property
from tqdm import tqdm
from sklearn.decomposition import PCA
import os

USE_IMAGES = True

csv_path = "data/properties.csv"

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

# ----------------------------------- IMAGE EMBEDDING -----------------------------------
if USE_IMAGES:
    print("Features before images:", df_clean.shape[1])
    emb_path = "data/image_embeddings_all.npy"
    has_path = "data/has_images.npy"
    refs_path = "data/image_refs.npy"

    reset = False # set to True to recompute embeddings from scratch
    if reset or not (os.path.exists(emb_path) and os.path.exists(has_path) and os.path.exists(refs_path)):
        emb_list = []
        has_images = []
        ref_arr = []

        for ref in tqdm(df_clean["reference"], total=len(df_clean), desc="Embedding property images"):
            emb = embed_property(ref, images_root="data/images")
            ref_arr.append(ref)

            if emb is None:
                emb_list.append(None)
                has_images.append(0)
            else:
                emb_list.append(emb)
                has_images.append(1)

        first_emb = next(e for e in emb_list if e is not None)
        emb_dim = first_emb.shape[0]

        emb_matrix = np.vstack([
            e if e is not None else np.zeros(emb_dim, dtype=np.float32)
            for e in emb_list
        ])

        np.save(emb_path, emb_matrix)
        np.save(has_path, np.array(has_images))
        np.save(refs_path, np.array(ref_arr))

    else:
        emb_matrix = np.load(emb_path)
        has_images = np.load(has_path)
        ref_arr = np.load(refs_path)

        ref_to_i = {ref: i for i, ref in enumerate(ref_arr)}

        emb_dim = emb_matrix.shape[1]
        aligned_embs = np.zeros((len(df_clean), emb_dim), dtype=np.float32)
        aligned_has = np.zeros(len(df_clean), dtype=np.int8)

        for j, ref in enumerate(df_clean["reference"].values):
            i = ref_to_i.get(ref)
            if i is not None:
                aligned_embs[j] = emb_matrix[i]
                aligned_has[j] = has_images[i]
            # else: keep zeros + has=0

        emb_matrix = aligned_embs
        has_images_arr = aligned_has

    has_images = np.array(has_images, dtype=np.int8)

    # add the flag into df_clean so it goes through encoding normally
    df_clean["has_images"] = has_images

    print("Embeddings shape (all rows):", emb_matrix.shape)
    print("Rows with images:", has_images.sum(), "/", len(has_images))

    print("Found embeddings for ", emb_matrix.shape[0], " properties.")



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

if USE_IMAGES:
    X_train, X_test, y_train, y_test, emb_train, emb_test = train_test_split(
        X, y, emb_matrix,
        test_size=0.2,
        random_state=42
    )

    pca = PCA(n_components=50, random_state=0)
    emb_train_red = pca.fit_transform(emb_train)
    emb_test_red  = pca.transform(emb_test)

    emb_cols = [f"img_{i}" for i in range(emb_train_red.shape[1])]

    X_train = pd.concat(
        [X_train.reset_index(drop=True),
        pd.DataFrame(emb_train_red, columns=emb_cols)],
        axis=1
    )

    X_test = pd.concat(
        [X_test.reset_index(drop=True),
        pd.DataFrame(emb_test_red, columns=emb_cols)],
        axis=1
    )

else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

tree = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=5,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=0,
    n_jobs=-1
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.0f} €")
print(f"R²: {r2:.3f}")