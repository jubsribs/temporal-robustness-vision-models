from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import PROCESSED_DIR, CAMERAS


def train_batch(camera):
    data_dir = PROCESSED_DIR / camera
    csvs = sorted(data_dir.glob("week_*.csv"))

    if len(csvs) < 2:
        print(f"[BATCH] {camera}: dados insuficientes")
        return

    test_csv = csvs[-1]
    train_csvs = csvs[:-1]

    train_dfs = []
    for csv in train_csvs:
        df = pd.read_csv(csv)
        if "ocupada" in df.columns:
            train_dfs.append(df)

    test_df = pd.read_csv(test_csv)

    if not train_dfs or test_df.empty:
        print(f"[BATCH] {camera}: erro nos dados")
        return

    train_df = pd.concat(train_dfs, ignore_index=True)

    X_train = train_df.drop(columns=["ocupada", "timestamp"]).fillna(0)
    y_train = train_df["ocupada"]

    X_test = test_df.drop(columns=["ocupada", "timestamp"]).fillna(0)
    y_test = test_df["ocupada"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(
        f"[BATCH] {camera} | "
        f"train_weeks={len(train_csvs)} | "
        f"train_samples={len(X_train)} | "
        f"test_week={test_csv.stem} | "
        f"test_samples={len(X_test)} | "
        f"acc={acc:.3f} | "
        f"f1={f1:.3f} | "
        f"recall={recall:.3f}"
    )


if __name__ == "__main__":
    for cam in CAMERAS:
        train_batch(cam)
