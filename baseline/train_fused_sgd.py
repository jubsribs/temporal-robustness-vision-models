import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle
import os

from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    classification_report,
)

from config import RESULTS_DIR

# Diretórios
FUSED_DIR = Path("data/fused")
FIG_DIR = Path("analysis/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_fused_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)

    if "ocupada" not in df.columns:
        raise ValueError(f"{csv_path.name} não tem coluna 'ocupada'")

    df["ocupada"] = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)
    return df


def _is_bad_metric(x: float) -> bool:
    # “mau” = 0 (numerador zero) OU NaN (indefinição/denominador zero)
    if x is None:
        return True
    try:
        if np.isnan(x):
            return True
    except Exception:
        pass
    return x == 0


def _value_or_nan(x: float) -> float:
    # Para barras: se for mau, mete NaN para a barra não aparecer
    return np.nan if _is_bad_metric(x) else float(x)


def train_fused_sgd(random_state: int = 42):
    csvs = sorted(FUSED_DIR.glob("week_*.csv"))
    if not csvs:
        print("[WARN] Nenhum CSV found em data/fused")
        return

    dfs = []
    for p in csvs:
        try:
            dfs.append(_load_fused_csv(p))
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")

    if not dfs:
        print("[WARN] Nenhum CSV fused válido para treino.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Features (numéricas) + target
    X = (
        df.drop(columns=["timestamp", "ocupada"], errors="ignore")
        .select_dtypes(include=["number"])
        .fillna(0)
    )
    y = df["ocupada"]

    if X.empty or y.empty:
        print("[WARN] Dataset vazio após pré-processamento.")
        return

    # split 80/20 global
    strat = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=strat,
    )

    model = SGDClassifier( loss="log_loss", random_state=42)

    # =========================
    # TEMPO + MEMÓRIA TREINO
    # =========================
    tracemalloc.start()
    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start_train
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # =========================
    # TEMPO PREDIÇÃO
    # =========================
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    # =========================
    # TAMANHO DO MODELO
    # =========================
    tmp_model = "rf_temp.pkl"
    with open(tmp_model, "wb") as f:
        pickle.dump(model, f)

    model_size_mb = os.path.getsize(tmp_model) / (1024 * 1024)
    os.remove(tmp_model)

    print(
        f"[PERFORMANCE] "
        f"train_time={train_time:.4f}s | "
        f"predict_time={predict_time:.4f}s | "
        f"peak_memory={peak_memory / (1024*1024):.2f}MB | "
        f"model_size={model_size_mb:.2f}MB"
    )

    # Métricas globais
    acc = accuracy_score(y_test, y_pred)

    # Métricas por classe (1 = Ocupado, 0 = Ausência/Livre)
    # zero_division=np.nan para capturar indefinições
    prec_occ = precision_score(y_test, y_pred, pos_label=1, zero_division=np.nan)
    rec_occ = recall_score(y_test, y_pred, pos_label=1, zero_division=np.nan)
    f1_occ = f1_score(y_test, y_pred, pos_label=1, zero_division=np.nan)

    prec_abs = precision_score(y_test, y_pred, pos_label=0, zero_division=np.nan)
    rec_abs = recall_score(y_test, y_pred, pos_label=0, zero_division=np.nan)
    f1_abs = f1_score(y_test, y_pred, pos_label=0, zero_division=np.nan)

    print(
        f"[FUSED SGD 80/20] samples={len(df)} | "
        f"train={len(X_train)} | test={len(X_test)} | acc={acc:.3f}"
    )

    # Report (para suportes)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Absence(0)", "Occupied(1)"],
        zero_division=np.nan,
        output_dict=True,
    )

    metrics_raw = {
        "split": "80/20",
        "random_state": random_state,
        "samples_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "peak_memory_MB": peak_memory / (1024 * 1024),
        "model_size_MB": model_size_mb,
        "accuracy": acc,
        "precision_occupied": prec_occ,
        "recall_occupied": rec_occ,
        "f1_occupied": f1_occ,
        "precision_absence": prec_abs,
        "recall_absence": rec_abs,
        "f1_absence": f1_abs,
        "support_absence": int(report["Absence(0)"]["support"]) if "Absence(0)" in report else None,
        "support_occupied": int(report["Occupied(1)"]["support"]) if "Occupied(1)" in report else None,
    }

    # ✅ CSV apenas com números (sem “◯”)
    pd.DataFrame([metrics_raw]).to_csv(
        RESULTS_DIR / "fused_sgd_metrics.csv",
        index=False,
    )
    print("[OK] Métricas salvas em data/results/fused_sgd_metrics.csv")

    # =========================
    # GRÁFICO DE MÉTRICAS (com “círculo oco” quando 0/NaN)
    # =========================
    plot_vals_raw = {
        "accuracy": acc,
        "prec_occupied": prec_occ,
        "rec_occupied": rec_occ,
        "f1_occupied": f1_occ,
        "prec_absence": prec_abs,
        "rec_absence": rec_abs,
        "f1_absence": f1_abs,
    }

    labels = list(plot_vals_raw.keys())
    raw_values = [plot_vals_raw[k] for k in labels]
    values = [_value_or_nan(v) for v in raw_values]  # NaN onde for mau

    plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))

    plt.bar(x, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Stochastic Gradient Descent — Fused (80/20)")
    plt.xticks(x, labels, rotation=30, ha="right")

    # desenha círculo oco por cima do label quando a métrica for 0/NaN
    for i, v in enumerate(raw_values):
        if _is_bad_metric(v):
            plt.scatter(
                [i],
                [0.05],            # altura fixa para ficar visível (ajusta se quiseres)
                marker="o",
                facecolors="none", # círculo oco
            )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_metrics_bar_sgd.png")
    plt.close()

    # =========================
    # MATRIZ DE CONFUSÃO
    # =========================
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Absence(0)", "Occupied(1)"],
        cmap="Blues",
        normalize=None,
    )
    disp.ax_.set_title("Confusion Matrix — Fused Stochastic Gradient Descent (80/20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix_sgd.png")
    plt.close()

    print("[OK] Gráficos salvos em analysis/figures")


if __name__ == "__main__":
    train_fused_sgd()
