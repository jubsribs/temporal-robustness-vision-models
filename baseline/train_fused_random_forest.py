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
from sklearn.model_selection import train_test_split

# Diretórios
FUSED_DIR = Path("data/fused")
RESULTS_DIR = Path("data/results")
FIG_DIR = Path("analysis/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def train_fused_random_forest():
    csvs = sorted(FUSED_DIR.glob("week_*.csv"))

    if len(csvs) < 2:
        print("[WARN] Dados insuficientes para treino")
        return

    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        if "ocupada" in df.columns:
            dfs.append(df)

    if not dfs:
        print("[WARN] Nenhum dado válido encontrado")
        return

    df = pd.concat(dfs, ignore_index=True)

    X = (
    df
    .drop(columns=["timestamp", "ocupada"], errors="ignore")
    .select_dtypes(include=["number"])
    .fillna(0)
    )

    y = df["ocupada"]

    if X.empty or y.empty:
        print("[WARN] Dataset vazio após pré-processamento")
        return



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,      # 20% teste
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Modelo
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
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(
        f"[FUSED RF] "
        f"samples={len(df)} | "
        f"train={len(X_train)} | "
        f"test={len(X_test)} | "
        f"acc={acc:.3f} | "
        f"f1={f1:.3f} | "
        f"recall={recall:.3f} | "
        f"precision={precision:.3f}"
    )

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
