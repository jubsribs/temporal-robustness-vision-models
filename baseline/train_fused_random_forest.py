from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

# Diretórios
FUSED_DIR = Path("data/fused")
RESULTS_DIR = Path("data/results")
FIG_DIR = Path("analysis/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def train_fused_random_forest():
    csvs = sorted(FUSED_DIR.glob("week_*_alpha_beta.csv"))

    if len(csvs) < 2:
        print("[WARN] Dados insuficientes para treino")
        return

    # split temporal
    test_csv = csvs[-1]
    train_csvs = csvs[:-1]

    train_df = pd.concat(
        [pd.read_csv(c) for c in train_csvs],
        ignore_index=True
    )

    test_df = pd.read_csv(test_csv)

    # Features e alvo
    X_train = train_df.drop(columns=["timestamp", "ocupada"]).fillna(0)
    y_train = train_df["ocupada"]

    X_test = test_df.drop(columns=["timestamp", "ocupada"]).fillna(0)
    y_test = test_df["ocupada"]

    # Modelo
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_week": test_csv.stem
    }

    # Salva CSV
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "fused_random_forest_metrics.csv",
        index=False
    )

    print("[OK] Métricas salvas em CSV")

    # =========================
    # 📊 GRÁFICO DE MÉTRICAS
    # =========================
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["accuracy", "precision", "recall", "f1"],
        [metrics[m] for m in ["accuracy", "precision", "recall", "f1"]]
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Random Forest — Fused Alpha + Beta")
    plt.tight_layout()

    plt.savefig(FIG_DIR / "fused_metrics_bar.png")
    plt.close()

    # =========================
    # 🔲 MATRIZ DE CONFUSÃO
    # =========================
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Livre", "Ocupado"],
        cmap="Blues",
        normalize=None
    )
    disp.ax_.set_title("Confusion Matrix — Fused Random Forest")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix.png")
    plt.close()

    print("[OK] Gráficos salvos em analysis/figures")


if __name__ == "__main__":
    train_fused_random_forest()
