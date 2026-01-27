from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from config import PROCESSED_DIR,RESULTS_DIR

CAMERAS = ["camera_alpha", "camera_beta"]
FIG_DIR = Path("analysis/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def month_id(week_id: str) -> str:
    # week_YYYY_WW → YYYY-MM
    _, year, week = week_id.split("_")
    return f"{year}-{int(week):02d}"


def train_camera_monthly(camera):
    data_dir = PROCESSED_DIR / camera
    csvs = sorted(data_dir.glob("week_*.csv"))

    if not csvs:
        print(f"[WARN] {camera}: sem dados")
        return []

    # agrupa semanas por mês
    monthly = {}
    for csv in csvs:
        mid = month_id(csv.stem)
        monthly.setdefault(mid, []).append(csv)

    results = []

    for month, files in sorted(monthly.items()):
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if "ocupada" not in df.columns:
            continue

        X = df.drop(columns=["ocupada", "timestamp"]).fillna(0)
        y = df["ocupada"]

        # split mensal
        strat = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        results.append({
            "camera": camera,
            "month": month,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,
            "f1": f1,
            "recall": recall
        })

        print(
            f"[MONTHLY] {camera} {month} | "
            f"train={len(X_train)} | test={len(X_test)} | "
            f"acc={acc:.3f} | f1={f1:.3f} | recall={recall:.3f}"
        )

    return results


def plot_metric(df, camera, metric):
    cam_df = df[df["camera"] == camera].sort_values("month")

    plt.figure()
    plt.plot(cam_df["month"], cam_df[metric], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("Mês")
    plt.title(f"{camera} — {metric} week")
    plt.tight_layout()

    plt.savefig(FIG_DIR / f"{camera}_{metric}.png")
    plt.close()


if __name__ == "__main__":
    all_results = []

    for cam in CAMERAS:
        all_results.extend(train_camera_monthly(cam))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_DIR / "monthly_rf.csv", index=False)

        for cam in CAMERAS:
            plot_metric(df, cam, "accuracy")
            plot_metric(df, cam, "f1")
            plot_metric(df, cam, "recall")
