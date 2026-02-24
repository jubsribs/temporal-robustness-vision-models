import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle
import os

from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
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


# Diretórios
FUSED_DIR = Path("data/fused")
BASE_DIR = Path("analysis/sgd_fused")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = BASE_DIR / timestamp

FIG_DIR = RUN_DIR / "figures"
METRICS_DIR = RUN_DIR / "metrics"
MODEL_DIR = RUN_DIR / "model"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sgd_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(loss="log_loss", class_weight="balanced"))
])

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

    model = SGDClassifier( loss="log_loss",
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,)

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

    # ======================================================
    # ANÁLISE FALSOS NEGATIVOS
    # ======================================================

    results_df = X_test.copy()
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = y_pred

    false_negatives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 0)
    ]

    true_positives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 1)
    ]

    if len(false_negatives) > 0 and len(true_positives) > 0:
        comparison = pd.DataFrame({
            "FN_mean": false_negatives.mean(numeric_only=True),
            "TP_mean": true_positives.mean(numeric_only=True),
        })

        comparison["difference"] = (
            comparison["FN_mean"] - comparison["TP_mean"]
        )

        comparison.sort_values(
            "difference",
            key=lambda x: np.abs(x),
            ascending=False
        ).to_csv(
            METRICS_DIR / "false_negative_analysis.csv"
        )

        plt.figure(figsize=(6,4))
        plt.hist(true_positives["average_light"], bins=30, alpha=0.5, label="TP")
        plt.hist(false_negatives["average_light"], bins=30, alpha=0.5, label="FN")
        plt.legend()
        plt.title("light em FN vs light em TP")
        plt.show()
        plt.savefig(METRICS_DIR / "fused_metrics_sgd.png")
        plt.close()

    # =========================
    # TAMANHO DO MODELO
    # =========================
    model_path = MODEL_DIR / "sgd_model.pkl"
    with open(tmp_model, "wb") as f:
        pickle.dump(model, f)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

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
        METRICS_DIR / "fused_sgd_metrics.csv",
        index=False,
    )
    print("[OK] Métricas salvas em", METRICS_DIR)

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
        labels=[0,1],
        display_labels=["Absence(0)", "Occupied(1)"],
        cmap="Blues",
        normalize=None,
    )
    disp.ax_.set_title("Confusion Matrix — Fused Stochastic Gradient Descent (80/20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix_sgd.png")
    plt.close()

    print("[OK] Gráficos salvos", FIG_DIR)


if __name__ == "__main__":
    train_fused_sgd()
