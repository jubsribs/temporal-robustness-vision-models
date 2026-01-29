from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

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


def week_sort_key(week_str: str):
    # "2026-W03" -> (2026, 3)
    try:
        y, w = week_str.split("-W")
        return (int(y), int(w))
    except Exception:
        return (9999, 99)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = (
        df.drop(columns=["ocupada", "timestamp"], errors="ignore")
          .select_dtypes(include=["number"])
          .fillna(0)
    )
    X.columns = X.columns.astype(str)
    return X


def get_feature_space(csvs) -> list[str]:
    # União de todas as features numéricas ao longo das semanas
    cols = set()
    for csv in csvs:
        df = pd.read_csv(csv)
        if "ocupada" not in df.columns:
            continue
        X = build_X(df)
        cols.update(X.columns.tolist())
    return sorted(cols)


def align_X(X: pd.DataFrame, feature_space: list[str]) -> pd.DataFrame:
    return X.reindex(columns=feature_space, fill_value=0)


def metric_value(y_true, y_pred, which: str, pos_label: int):
    # zero_division=np.nan => métricas indefinidas ficam NaN
    if which == "precision":
        return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    if which == "recall":
        return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    if which == "f1":
        return f1_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    raise ValueError("which inválido")


def display_or_circle(x):
    if x is None:
        return "◯"
    try:
        if np.isnan(x):
            return "◯"
    except Exception:
        pass
    if x == 0:
        return "◯"
    return f"{float(x):.4f}"


def plot_metric(df: pd.DataFrame, camera: str, metric: str, suffix: str):
    cam_df = df[df["camera"] == camera].copy()
    cam_df["__k"] = cam_df["week"].apply(week_sort_key)
    cam_df = cam_df.sort_values("__k").drop(columns="__k")

    y_raw = cam_df[metric].astype(float).to_numpy()
    invalid_mask = np.isnan(y_raw) | (y_raw == 0)

    y_line = y_raw.copy()
    y_line[invalid_mask] = np.nan

    x = cam_df["week"].to_numpy()

    plt.figure()
    plt.plot(x, y_line, marker="o")

    if invalid_mask.any():
        x_bad = x[invalid_mask]
        y_bad = np.full(len(x_bad), 0.02)
        plt.scatter(x_bad, y_bad, marker="o", facecolors="none")

    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel("Week")
    plt.title(f"{camera} — {suffix} (incremental SGD)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{camera}_{suffix}_sgd_incremental.png")
    plt.close()


def train_camera_incremental(camera: str):
    data_dir = PROCESSED_DIR / camera
    csvs = sorted(data_dir.glob("week_*.csv"))

    if not csvs:
        print(f"[WARN] {camera}: sem dados")
        return []

    feature_space = get_feature_space(csvs)
    if not feature_space:
        print(f"[WARN] {camera}: sem features numéricas")
        return []

    print(f"[INFO] {camera}: espaço comum com {len(feature_space)} features")

    model = SGDClassifier(loss="log_loss", random_state=42)
    model_initialized = False

    results = []

    for csv in csvs:
        df = pd.read_csv(csv)

        if "ocupada" not in df.columns:
            print(f"[SKIP] {camera} {csv.name}: sem coluna 'ocupada'")
            continue

        X = align_X(build_X(df), feature_space)
        y = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)

        if X.empty or y.empty:
            print(f"[SKIP] {camera} {csv.name}: dataset vazio após pré-processamento")
            continue

        # split dentro da semana (para métricas dessa semana)
        strat = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )

        week_id = week_id_from_stem(csv.stem)

        # Treino incremental
        if not model_initialized:
            # primeira semana: temos de passar "classes"
            model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
            model_initialized = True
        else:
            model.partial_fit(X_train, y_train)

        # Avaliação na semana (no holdout dessa semana)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # Ocupado (1)
        prec_occ = metric_value(y_test, y_pred, "precision", pos_label=1)
        rec_occ  = metric_value(y_test, y_pred, "recall", pos_label=1)
        f1_occ   = metric_value(y_test, y_pred, "f1", pos_label=1)

        # Ausência (0)
        prec_abs = metric_value(y_test, y_pred, "precision", pos_label=0)
        rec_abs  = metric_value(y_test, y_pred, "recall", pos_label=0)
        f1_abs   = metric_value(y_test, y_pred, "f1", pos_label=0)

        results.append({
            "camera": camera,
            "week": week_id,
            "file": csv.name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,

            "precision_occupied": prec_occ,
            "recall_occupied": rec_occ,
            "f1_occupied": f1_occ,

            "precision_absence": prec_abs,
            "recall_absence": rec_abs,
            "f1_absence": f1_abs,

            "display_precision_occupied": display_or_circle(prec_occ),
            "display_recall_occupied": display_or_circle(rec_occ),
            "display_f1_occupied": display_or_circle(f1_occ),
            "display_precision_absence": display_or_circle(prec_abs),
            "display_recall_absence": display_or_circle(rec_abs),
            "display_f1_absence": display_or_circle(f1_abs),
        })

        print(
            f"[INCR] {camera} {week_id} | train={len(X_train)} test={len(X_test)} "
            f"| acc={acc:.3f} | occ(p={prec_occ} r={rec_occ} f1={f1_occ}) "
            f"| abs(p={prec_abs} r={rec_abs} f1={f1_abs})"
        )

    return results


if __name__ == "__main__":
    all_results = []

    for cam in CAMERAS:
        all_results.extend(train_camera_incremental(cam))

    if all_results:
        df = pd.DataFrame(all_results)

        out_csv = RESULTS_DIR / "weekly_sgd_incremental_with_absence.csv"
        df.to_csv(out_csv, index=False)

        for cam in CAMERAS:
            plot_metric(df, cam, "accuracy", "accuracy")

            plot_metric(df, cam, "precision_occupied", "precision_occupied")
            plot_metric(df, cam, "recall_occupied", "recall_occupied")
            plot_metric(df, cam, "f1_occupied", "f1_occupied")

            plot_metric(df, cam, "precision_absence", "precision_absence")
            plot_metric(df, cam, "recall_absence", "recall_absence")
            plot_metric(df, cam, "f1_absence", "f1_absence")

        print(f"[OK] Resultados: {out_csv} | Figuras: {FIG_DIR}/")
    else:
        print("[WARN] Nenhum resultado gerado")
