from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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


def _display_or_circle(x: float) -> str:
    # Marca com círculo quando:
    # - indefinido (NaN)
    # - ou valor exatamente 0 (numerador zero, conforme pedido)
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return "◯"
    return f"{x:.4f}"


def _plot_value_or_nan(x: float) -> float:
    # Omitir do gráfico se for 0 ou NaN
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return np.nan
    return float(x)


def train_fused_random_forest(random_state: int = 42):
    csvs = sorted(FUSED_DIR.glob("week_*.csv"))
    if not csvs:
        print("[WARN] Nenhum CSV found em data/fused")
        return

    # Carrega tudo (já fused)
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
    print(f" features treinadas com a tabela",X)
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

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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
        f"[FUSED RF 80/20] samples={len(df)} | "
        f"train={len(X_train)} | test={len(X_test)} | "
        f"acc={acc:.3f}"
    )
    print(
        f"  Ocupado(1): precision={prec_occ if not np.isnan(prec_occ) else 'NaN'} | "
        f"recall={rec_occ if not np.isnan(rec_occ) else 'NaN'} | "
        f"f1={f1_occ if not np.isnan(f1_occ) else 'NaN'}"
    )
    print(
        f"  Ausência(0): precision={prec_abs if not np.isnan(prec_abs) else 'NaN'} | "
        f"recall={rec_abs if not np.isnan(rec_abs) else 'NaN'} | "
        f"f1={f1_abs if not np.isnan(f1_abs) else 'NaN'}"
    )

    # Report completo (opcional mas útil)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Absence(0)", "Occupied(1)"],
        zero_division=np.nan,
        output_dict=True,
    )

    # Métricas (raw)
    metrics_raw = {
        "split": "80/20",
        "random_state": random_state,
        "samples_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "accuracy": acc,
        # Ocupado (1)
        "precision_ocupado": prec_occ,
        "recall_ocupado": rec_occ,
        "f1_ocupado": f1_occ,
        # Ausência (0)
        "precision_ausencia": prec_abs,
        "recall_ausencia": rec_abs,
        "f1_ausencia": f1_abs,
        # suporte (para contexto)
        "support_ausencia": int(report["Absence(0)"]["support"]) if "Absence(0)" in report else None,
        "support_ocupado": int(report["Occupied(1)"]["support"]) if "Occupied(1)" in report else None,
    }

    # Métricas para “display” (com ◯)
    metrics_display = {k: _display_or_circle(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
                       for k, v in metrics_raw.items()}

    # Salva CSV (raw + display lado a lado)
    out_df = pd.DataFrame([metrics_raw])
    out_df_display = pd.DataFrame([metrics_display]).add_prefix("display__")
    pd.concat([out_df, out_df_display], axis=1).to_csv(
        RESULTS_DIR / "fused_random_forest_metrics.csv",
        index=False,
    )
    print("[OK] Métricas salvas em data/results/fused_random_forest_metrics.csv")

    # =========================
    # GRÁFICOS (omitindo 0 e NaN)
    # =========================
    # 1) barras: accuracy + métricas por classe
    plot_vals = {
        "accuracy": _plot_value_or_nan(acc),
        "prec_ocupado": _plot_value_or_nan(prec_occ),
        "rec_ocupado": _plot_value_or_nan(rec_occ),
        "f1_ocupado": _plot_value_or_nan(f1_occ),
        "prec_ausencia": _plot_value_or_nan(prec_abs),
        "rec_ausencia": _plot_value_or_nan(rec_abs),
        "f1_ausencia": _plot_value_or_nan(f1_abs),
    }

    labels = list(plot_vals.keys())
    values = [plot_vals[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Random Forest — Fused (80/20) — (0 e indefinidos omitidos)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_metrics_bar.png")
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
    disp.ax_.set_title("Confusion Matrix — Fused RF (80/20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix.png")
    plt.close()

    print("[OK] Gráficos salvos em analysis/figures")


if __name__ == "__main__":
    train_fused_random_forest()
