from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import SGDClassifier
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


def metric_value(y_true, y_pred, which: str, pos_label: int):
    """
    which: 'precision' | 'recall' | 'f1'
    pos_label: 1 (ocupado) ou 0 (ausência)
    Regras:
      - se der indefinido (denominador zero) -> NaN
      - se der 0 (numerador zero) -> 0
    """
    if which == "precision":
        return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    if which == "recall":
        return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    if which == "f1":
        return f1_score(y_true, y_pred, pos_label=pos_label, zero_division=np.nan)
    raise ValueError("which inválido")


def display_or_circle(x):
    # Círculo se:
    # - NaN (indefinido/denominador zero)
    # - ou 0 (numerador zero)
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


def plot_value_or_nan(x):
    # Omitir do gráfico se 0 ou NaN
    if x is None:
        return np.nan
    try:
        if np.isnan(x):
            return np.nan
    except Exception:
        pass
    if x == 0:
        return np.nan
    return float(x)


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
        print(f" features treinadas com a tabela", X)
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

        model = SGDClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        week_id = week_id_from_stem(csv.stem)

        # Métrica global
        acc = accuracy_score(y_test, y_pred)

        # Métricas para Ocupado (1)
        prec_occ = metric_value(y_test, y_pred, "precision", pos_label=1)
        rec_occ = metric_value(y_test, y_pred, "recall", pos_label=1)
        f1_occ = metric_value(y_test, y_pred, "f1", pos_label=1)

        # Métricas para Ausência (0)
        prec_abs = metric_value(y_test, y_pred, "precision", pos_label=0)
        rec_abs = metric_value(y_test, y_pred, "recall", pos_label=0)
        f1_abs = metric_value(y_test, y_pred, "f1", pos_label=0)

        results.append(
            {
                "camera": camera,
                "week": week_id,
                "file": csv.name,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": acc,

                # Ocupado (1)
                "precision_occupied": prec_occ,
                "recall_occupied": rec_occ,
                "f1_occupied": f1_occ,

                # Ausência (0)
                "precision_absence": prec_abs,
                "recall_absence": rec_abs,
                "f1_absence": f1_abs,

                # “display” (◯ quando 0 ou indefinido)
                "display_precision_occupied": display_or_circle(prec_occ),
                "display_recall_occupied": display_or_circle(rec_occ),
                "display_f1_occupied": display_or_circle(f1_occ),
                "display_precision_absence": display_or_circle(prec_abs),
                "display_recall_absence": display_or_circle(rec_abs),
                "display_f1_absence": display_or_circle(f1_abs),
            }
        )

        print(
            f"[WEEKLY] {camera} {week_id} | "
            f"train={len(X_train)} | test={len(X_test)} | acc={acc:.3f} | "
            f"occ(p={prec_occ if not np.isnan(prec_occ) else 'NaN'} "
            f"r={rec_occ if not np.isnan(rec_occ) else 'NaN'} "
            f"f1={f1_occ if not np.isnan(f1_occ) else 'NaN'}) | "
            f"abs(p={prec_abs if not np.isnan(prec_abs) else 'NaN'} "
            f"r={rec_abs if not np.isnan(rec_abs) else 'NaN'} "
            f"f1={f1_abs if not np.isnan(f1_abs) else 'NaN'})"
        )

    return results


def plot_metric(df: pd.DataFrame, camera: str, metric: str, suffix: str):
    """
    metric: coluna numérica a plotar
    suffix: nome no ficheiro
    Regras:
      - valores válidos: traça linha + marcadores normais
      - valores 0 ou NaN: desenha um círculo oco no gráfico (marker circular)
    """
    cam_df = df[df["camera"] == camera].sort_values("week").copy()

    # y original
    y_raw = cam_df[metric].astype(float).to_numpy()

    # inválidos: 0 ou NaN
    invalid_mask = np.isnan(y_raw) | (y_raw == 0)

    # linha: inválidos viram NaN (para a linha quebrar nesses pontos)
    y_line = y_raw.copy()
    y_line[invalid_mask] = np.nan

    x = cam_df["week"].to_numpy()

    plt.figure()

    # Linha com pontos válidos
    plt.plot(x, y_line, marker="o")

    # Marcadores circulares (ocos) nos pontos inválidos
    if invalid_mask.any():
        x_bad = x[invalid_mask]
        # coloca o círculo ligeiramente acima do 0 para ficar visível
        y_bad = np.full(len(x_bad), 0.02)

        plt.scatter(
            x_bad,
            y_bad,
            marker="o",
            facecolors="none",   # círculo oco
        )

    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel("Week")
    plt.title(f"{camera} — {suffix} (weekly)")
    plt.tight_layout()

    plt.savefig(FIG_DIR / f"{camera}_{suffix}_weekly.png")
    plt.close()


if __name__ == "__main__":
    all_results = []

    for cam in CAMERAS:
        all_results.extend(train_camera_weekly(cam))

    if all_results:
        df = pd.DataFrame(all_results)

        # Guarda CSV com valores numéricos + colunas display_*
        df.to_csv(RESULTS_DIR / "weekly_rf_with_absence.csv", index=False)

        for cam in CAMERAS:
            # global
            plot_metric(df, cam, "accuracy", "accuracy")

            # occupied (1)
            plot_metric(df, cam, "precision_occupied", "precision_occupied")
            plot_metric(df, cam, "recall_occupied", "recall_occupied")
            plot_metric(df, cam, "f1_occupied", "f1_occupied")

            # ausência (0)
            plot_metric(df, cam, "precision_absence", "precision_absence")
            plot_metric(df, cam, "recall_absence", "recall_absence")
            plot_metric(df, cam, "f1_absence", "f1_absence")

        print("[OK] Resultados salvos em data/results/weekly_rf_with_absence.csv e figuras em analysis/figures/")
    else:
        print("[WARN] Nenhum resultado gerado")
