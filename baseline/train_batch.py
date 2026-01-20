from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from config import PROCESSED_DIR


def train_batch(camera):
    data_dir = PROCESSED_DIR / camera
    dfs = []

    for csv in sorted(data_dir.glob("week_*.csv")):
        df = pd.read_csv(csv)
        if "ocupada" in df.columns:
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    X = df.drop(columns=["ocupada", "timestamp"]).fillna(0)
    y = df["ocupada"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)

    print(f"[BATCH] {camera} F1={f1:.3f}")
