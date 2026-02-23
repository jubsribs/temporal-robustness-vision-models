from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, CAMERAS

OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = Path("analysis/feature_importance")
CSV_OUT.mkdir(parents=True, exist_ok=True)


def load_batch_data(camera):
    data_dir = PROCESSED_DIR / camera
    dfs = []

    for csv in sorted(data_dir.glob("week_*.csv")):
        df = pd.read_csv(csv)
        if "ocupada" in df.columns:
            dfs.append(df)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    X = df.drop(columns=["ocupada", "timestamp"]).fillna(0)
    y = df["ocupada"]

    return X, y


def compute_feature_importance(camera):
    data = load_batch_data(camera)
    if data is None:
        print(f"[WARN] {camera}: sem dados")
        return

    X, y = data

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    model.fit(X, y)

    coefs = model.named_steps["clf"].coef_[0]
    importance = np.abs(coefs)

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

    # Normaliza (opcional, mas recomendado)
    imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()

    # Salva CSV
    imp_df.to_csv(
        CSV_OUT / f"{camera}_feature_importance.csv",
        index=False
    )

    # Plot Top-10
    topk = imp_df.head(10)

    plt.figure(figsize=(8, 4))

    plt.bar(topk["feature"], topk["importance_norm"])

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Importance")
    plt.xlabel("Feature")
    plt.title(f"{camera} — Top 10 features")

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"{camera}_feature_importance.png")
    plt.close()

    print(f"[OK] Feature importance gerada para {camera}")


if __name__ == "__main__":
    for cam in CAMERAS:
        compute_feature_importance(cam)
