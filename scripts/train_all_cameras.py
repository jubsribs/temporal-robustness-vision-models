from config import CAMERAS
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_all():
    for cam in CAMERAS:
        print(f"Treinando {cam}...")

        data_dir = Path("data/processed") / cam
        model_dir = Path("models") / cam
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "rf_incremental.joblib"

        X_accum, y_accum = [], []
        results = []

        for csv in sorted(data_dir.glob("week_*.csv")):
            print(f"  → Semana {csv.stem}")

            df = pd.read_csv(csv, parse_dates=["timestamp"])

            if df.empty or "ocupada" not in df.columns:
                print(f"[WARN] CSV inválido: {csv}")
                continue

            X_week = df.drop(columns=["ocupada", "timestamp"])
            y_week = df["ocupada"]

            # remove linhas com NaN (sensores faltantes)
            mask = X_week.notna().all(axis=1)
            X_week, y_week = X_week[mask], y_week[mask]

            if X_week.empty:
                print(f"[WARN] Sem dados válidos após limpeza em {csv}")
                continue

            # acumula dados
            X_accum.append(X_week)
            y_accum.append(y_week)

            X_train = pd.concat(X_accum, ignore_index=True)
            y_train = pd.concat(y_accum, ignore_index=True)

            # carrega ou cria modelo
            if model_path.exists():
                model = joblib.load(model_path)
            else:
                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )

            # re-treina com histórico completo
            model.fit(X_train, y_train)

            # avalia SOMENTE na semana atual
            y_pred = model.predict(X_week)

            results.append({
                "camera": cam,
                "week": csv.stem,
                "train_samples": len(X_train),
                "test_samples": len(X_week),
                "accuracy": accuracy_score(y_week, y_pred),
                "f1": f1_score(y_week, y_pred)
            })

            joblib.dump(model, model_path)

        if results:
            out = Path("data/results")
            out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(results).to_csv(
                out / f"{cam}_weekly.csv",
                index=False
            )

if __name__ == "__main__":
    train_all()

