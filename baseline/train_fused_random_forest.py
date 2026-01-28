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
from config import RESULTS_DIR

# Diretórios
FUSED_DIR = Path("data/fused")
FIG_DIR = Path("analysis/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_fused_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)

    if "ocupada" not in df.columns:
        # Se acontecer, considera como inválido
        raise ValueError(f"{csv_path.name} não tem coluna 'ocupada'")

    # Normaliza target
    df["ocupada"] = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)
    return df


def train_fused_random_forest(test_week: str | None = None):
    csvs = sorted(FUSED_DIR.glob("week_*.csv"))

    if len(csvs) < 2:
        print("[WARN] Precisas de pelo menos 2 semanas fused para treinar/testar por semana.")
        return

    # Escolhe semana de teste (por defeito: última)
    if test_week is None:
        test_csv = csvs[-1]
    else:
        matches = [p for p in csvs if p.stem == test_week or p.name == test_week]
        if not matches:
            print(f"[WARN] test_week '{test_week}' não encontrado em {FUSED_DIR}")
            return
        test_csv = matches[0]

    train_csvs = [p for p in csvs if p != test_csv]

    # Carrega treino
    train_dfs = []
    for p in train_csvs:
        try:
            train_dfs.append(_load_fused_csv(p))
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")

    # Carrega teste
    try:
        df_test = _load_fused_csv(test_csv)
    except Exception as e:
        print(f"[WARN] Semana de teste inválida ({test_csv.name}): {e}")
        return

    if not train_dfs:
        print("[WARN] Nenhum CSV fused válido para treino.")
        return

    df_train = pd.concat(train_dfs, ignore_index=True)

    # Features: mantém apenas numéricas e remove timestamp/ocupada
    X_train = (
        df_train.drop(columns=["timestamp", "ocupada"], errors="ignore")
        .select_dtypes(include=["number"])
        .fillna(0)
    )
    y_train = df_train["ocupada"]

    X_test = (
        df_test.drop(columns=["timestamp", "ocupada"], errors="ignore")
        .select_dtypes(include=["number"])
        .fillna(0)
    )
    y_test = df_test["ocupada"]

    if X_train.empty or X_test.empty:
        print("[WARN] Features vazias após pré-processamento (confere colunas numéricas).")
        return

    # Alinha colunas (garante que train e test têm as mesmas features)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Modelo
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
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(
        f"[FUSED RF] "
        f"train_samples={len(df_train)} | "
        f"test_samples={len(df_test)} | "
        f"test_week={test_csv.stem} | "
        f"acc={acc:.3f} | "
        f"f1={f1:.3f} | "
        f"recall={recall:.3f} | "
        f"precision={precision:.3f}"
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "train_weeks": len(train_csvs),
        "test_week": test_csv.stem,
        "n_features": X_train.shape[1],
    }

    # Salva métricas
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "fused_random_forest_metrics.csv",
        index=False,
    )
    print("[OK] Métricas salvas em data/results/fused_random_forest_metrics.csv")

    # =========================
    # GRÁFICO DE MÉTRICAS
    # =========================
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["accuracy", "precision", "recall", "f1"],
        [metrics[m] for m in ["accuracy", "precision", "recall", "f1"]],
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Random Forest — Fused Alpha + Beta")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_metrics_bar.png")
    plt.close()

    # =========================
    # MATRIZ DE CONFUSÃO
    # =========================
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Livre", "Ocupado"],
        cmap="Blues",
        normalize=None,
    )
    disp.ax_.set_title(f"Confusion Matrix — Fused RF ({test_csv.stem})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix.png")
    plt.close()

    print("[OK] Gráficos salvos em analysis/figures")


if __name__ == "__main__":
    # por defeito usa a última semana como teste
    train_fused_random_forest()
    # se quiseres escolher:
    # train_fused_random_forest(test_week="week_2026_04")
