from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from config import PROCESSED_DIR, RESULTS_DIR

CAMERAS = ["camera_alpha", "camera_beta"]
FIG_DIR = Path("analysis/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def week_id_from_stem(stem: str) -> str:
    # Ex.: "week_2026_03" -> "2026-W03"
    parts = stem.split("_")
    if len(parts) != 3:
        return stem
    _, year, week = parts
    return f"{year}-W{int(week):02d}"


def train_camera_weekly(camera: str):
    data_dir = PROCESSED_DIR / camera
    csvs = sorted(data_dir.glob("week_*.csv"))

    if not csvs:
        print(f"[WARN] {camera}: sem dados")
        return []

    results = []

    for csv in csvs:
        df = pd.read_csv(csv)

        if "ocupada" not in df.columns:
            print(f"[SKIP] {camera} {csv.name}: sem coluna 'ocupada'")
            continue

        # Features: remove target e timestamp, mantém apenas numéricas
        X = (
            df.drop(columns=["ocupada", "timestamp"], errors="ignore")
            .select_dtypes(include=["number"])
            .fillna(0)
        )
        y = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)

        if X.empty or y.empty:
            print(f"[SKIP] {camera} {csv.name}: dataset vazio após pré-processamento")
            continue

        # split por semana (dentro do ficheiro)
        strat = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat,
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        week_id = week_id_from_stem(csv.stem)

        results.append(
            {
                "camera": camera,
                "week": week_id,
                "file": csv.name,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

        print(
            f"[WEEKLY] {camera} {week_id} | "
            f"train={len(X_train)} | test={len(X_test)} | "
            f"acc={acc:.3f} | f1={f1:.3f} | recall={recall:.3f} | precision={precision:.3f}"
        )

    return results


def plot_metric(df: pd.DataFrame, camera: str, metric: str):
    cam_df = df[df["camera"] == camera].sort_values("week")

    plt.figure()
    plt.plot(cam_df["week"], cam_df[metric], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("Week")
    plt.title(f"{camera} — {metric} (weekly)")
    plt.tight_layout()

    plt.savefig(FIG_DIR / f"{camera}_{metric}_weekly.png")
    plt.close()


if __name__ == "__main__":
    all_results = []

    for cam in CAMERAS:
        all_results.extend(train_camera_weekly(cam))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_DIR / "weekly_rf.csv", index=False)

        for cam in CAMERAS:
            plot_metric(df, cam, "accuracy")
            plot_metric(df, cam, "f1")
            plot_metric(df, cam, "precision")
            plot_metric(df, cam, "recall")

        print("[OK] Resultados salvos em data/results/weekly_rf.csv e figuras em analysis/figures/")
    else:
        print("[WARN] Nenhum resultado gerado")
