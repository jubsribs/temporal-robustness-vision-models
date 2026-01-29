from config import CAMERAS
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from datetime import date

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def current_week_id():
    year, week, _ = date.today().isocalendar()
    return f"week_{year}_{week:02d}"


def load_latest_model(model_dir: Path):
    models = sorted(model_dir.glob("week_*_incremental.joblib"))
    if not models:
        return None
    return joblib.load(models[-1])

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = (
        df.drop(columns=["timestamp", "ocupada"], errors="ignore")
          .select_dtypes(include=["number"])
          .fillna(0)
    )
    # garantir nomes consistentes (opcional, mas ajuda)
    X.columns = X.columns.astype(str)
    return X

def align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    # Se o modelo ainda não tem features guardadas, não alinha
    if not hasattr(model, "feature_names_in_"):
        return X

    expected = list(model.feature_names_in_)
    # Reindex: cria colunas em falta com 0 e descarta extras
    X_aligned = X.reindex(columns=expected, fill_value=0)
    return X_aligned


def train_all():
    week_id = current_week_id()
    print(f"Processando semana atual: {week_id}")

    for cam in CAMERAS:
        print(f"Treinando {cam}...")

        data_dir = Path("data/processed") / cam
        model_dir = Path("models") / cam
        model_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / f"{week_id}.csv"
        if not csv_path.exists():
            print(f"[INFO] Sem dados para {cam} em {week_id}")
            continue

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        if df.empty or "ocupada" not in df.columns:
            print(f"[WARN] CSV inválido: {csv_path}")
            continue

        X = build_X(df)
        y = df["ocupada"]

        if y.nunique() < 2:
            print("[WARN] Apenas uma classe presente. Pulando treino.")
            continue

        # Carrega modelo anterior
        model = load_latest_model(model_dir)

        if model is None:
            print("  → Criando modelo incremental inicial")
            model = SGDClassifier(loss="log_loss", random_state=42)

            # split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

        else:
            print("Atualizando modelo com dados de treino da semana")

            # Alinha X às features do modelo ANTES do split, para consistência total
            X = align_features_to_model(X, model)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model.partial_fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(
            f"  Resultado {week_id}: "
            f"train={len(X_train)}, "
            f"test={len(X_test)}, "
            f"acc={acc:.3f}, "
            f"f1={f1:.3f}"
        )

        # Salva modelo atualizado
        model_path = model_dir / f"{week_id}_incremental.joblib"
        joblib.dump(model, model_path)

        # Salva métricas
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([{
            "camera": cam,
            "week": week_id,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": acc,
            "f1": f1
        }]).to_csv(
            results_dir / f"{cam}_{week_id}.csv",
            index=False
        )


if __name__ == "__main__":
    train_all()