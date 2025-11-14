import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

# configuration
BASE = Path(r"c:\Users\knuser\Documents\secom")
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def find_files(base):
    files = {p.name.lower(): p for p in base.iterdir() if p.is_file()}
    return files

def try_read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        try:
            df = pd.read_csv(path, header=None)
            return df
        except Exception as e:
            print("Failed to read", path, e)
            return None

def load_data(base):
    files = find_files(base)
    print("Files found:", list(files.keys()))
    data_df = None
    labels = None

    # pick largest CSV/data file as data
    candidates = [files[n] for n in files if n.endswith(".csv") or n.endswith(".data")]
    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    if candidates:
        data_df = try_read_csv(candidates[0])
        print("Using data file:", candidates[0].name)

    # try common label filenames
    for name, path in files.items():
        if "label" in name or name.endswith(".labels") or "secom.labels" in name:
            labels = try_read_csv(path)
            print("Using label file:", path.name)
            break

    # fallback: any small single-column file may be labels
    if labels is None and len(candidates) > 1:
        for p in candidates[1:4]:
            dfp = try_read_csv(p)
            if dfp is not None and (dfp.shape[1] == 1 and dfp.shape[0] == candidates[0].stat().st_size):
                labels = dfp
                print("Fallback label file:", p.name)
                break

    return data_df, labels

def normalize_column_names(df):
    if df is None:
        return None
    # if header missing or non-descriptive, create generic names
    try:
        cols = list(df.columns)
        numeric_names = all(isinstance(c, int) or str(c).strip().isdigit() for c in cols)
    except Exception:
        numeric_names = True
    if numeric_names or any(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = [f"x{i}" for i in range(df.shape[1])]
    return df

def align_labels(X, y):
    if y is None:
        return X, None
    # collapse label df to single series
    if y.shape[1] > 1:
        # choose first numeric column if exists
        numeric_cols = [c for c in y.columns if pd.api.types.is_numeric_dtype(y[c])]
        lbl = y[numeric_cols[0]] if numeric_cols else y.iloc[:,0]
    else:
        lbl = y.iloc[:,0]
    # normalize common label conventions
    lbl = lbl.replace({-1:1, 1:0}) if set(lbl.dropna().unique()) <= {-1,1} else lbl
    # align lengths if equal
    if len(lbl) == len(X):
        return X.reset_index(drop=True), lbl.reset_index(drop=True)
    # otherwise try to broadcast or reindex safely
    print("Label length mismatch:", len(lbl), "vs data", len(X))
    lbl = lbl.reset_index(drop=True)
    lbl = lbl.reindex(range(len(X))).fillna(method="ffill").fillna(0)
    return X.reset_index(drop=True), lbl

def basic_eda(X, y):
    summary = {}
    summary['n_rows'], summary['n_cols'] = X.shape
    missing = X.isna().sum()
    summary['top_missing_percent'] = (missing / X.shape[0] * 100).sort_values(ascending=False).head(20).to_dict()
    summary['col_dtype_counts'] = X.dtypes.value_counts().to_dict()
    if y is not None:
        try:
            summary['label_distribution'] = pd.Series(y).value_counts(dropna=False).to_dict()
        except Exception:
            summary['label_distribution'] = {}
    pd.Series(summary).to_json(OUT / "eda_summary.json")
    missing.to_csv(OUT / "missing_counts.csv")
    return summary

def impute_scale(X):
    num = X.select_dtypes(include=[np.number])
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(num), columns=num.columns, index=num.index)
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(Xi), columns=Xi.columns, index=Xi.index)
    return Xi, Xs

def pca_and_plot(Xs):
    pca = PCA(n_components=min(50, Xs.shape[1]), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    pd.Series(evr).to_csv(OUT / "pca_explained_variance_ratio.csv")
    plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(evr)[:50], marker='o')
    plt.xlabel("n components"); plt.ylabel("cumulative explained variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "pca_cumulative.png", dpi=150)
    plt.close()
    return pca, pcs

def umap_plot(Xs, labels=None):
    try:
        import umap
    except Exception:
        print("umap not installed; skipping UMAP")
        return
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(Xs)
    plt.figure(figsize=(6,5))
    if labels is None:
        plt.scatter(emb[:,0], emb[:,1], s=6, alpha=0.6)
    else:
        lab = pd.Series(labels).fillna("na").astype(str)
        sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=lab, palette="tab10", s=8, alpha=0.8, legend="brief")
        plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.title("UMAP 2D")
    plt.tight_layout()
    plt.savefig(OUT / "umap2d.png", dpi=150)
    plt.close()

def isolation_outliers(Xs, top_k=50):
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=RANDOM_STATE)
    iso.fit(Xs)
    scores = iso.decision_function(Xs)
    df = pd.DataFrame({'iso_score': -scores}, index=Xs.index)
    df.sort_values("iso_score", ascending=False).head(top_k).to_csv(OUT / "top_iso_anomalies.csv")
    return df

def supervised_eval(Xi, y):
    if y is None:
        return None
    ynum = pd.to_numeric(pd.Series(y).astype(float), errors='coerce').fillna(0).astype(int)
    # normalize binary labels if necessary
    unique = set(ynum.unique())
    if not unique.issubset({0,1}):
        ynum = (ynum != ynum.mode()[0]).astype(int)
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    try:
        scores['f1_cv'] = cross_val_score(clf, Xi, ynum, cv=skf, scoring="f1").mean()
        scores['roc_auc_cv'] = cross_val_score(clf, Xi, ynum, cv=skf, scoring="roc_auc").mean()
        scores['pr_auc_cv'] = cross_val_score(clf, Xi, ynum, cv=skf, scoring="average_precision").mean()
    except Exception as e:
        print("Supervised evaluation failed:", e)
        return None
    clf.fit(Xi, ynum)
    fi = pd.Series(clf.feature_importances_, index=Xi.columns).sort_values(ascending=False)
    fi.head(100).to_csv(OUT / "feature_importances_top100.csv")
    return scores

def main():
    Xraw, yraw = load_data(BASE)
    if Xraw is None:
        print("No data file found in", BASE)
        return
    Xraw = normalize_column_names(Xraw)
    X, y = align_labels(Xraw, yraw)
    print("Data shape:", X.shape)
    summary = basic_eda(X, y)
    Xi, Xs = impute_scale(X)
    pca, pcs = pca_and_plot(Xs)
    umap_plot(Xs, y if y is not None else None)
    iso_df = isolation_outliers(Xs)
    iso_df.to_csv(OUT / "isolation_scores_all.csv")
    sup = supervised_eval(Xi, y)
    pd.DataFrame({"summary": list(summary.keys()), "value": list(map(str, summary.values()))}).to_csv(OUT / "summary_table.csv", index=False)
    if sup:
        pd.Series(sup).to_csv(OUT / "supervised_scores.csv")
    print("Outputs written to", OUT)

if __name__ == "__main__":
    main()
