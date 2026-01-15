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

        results = []

        for csv in sorted(data_dir.glob("week_*.csv")):
            df = pd.read_csv(csv, parse_dates=["timestamp"])

            if df.empty:
                print(f"[WARN] CSV vazio: {csv}")
                continue

            if "ocupada" not in df.columns:
                print(f"[WARN] Sem coluna 'ocupada' em {csv}")
                continue

            X = df.drop(columns=["ocupada", "timestamp"])
            y = df["ocupada"]

            # remove linhas com NaN (sensores faltantes)
            mask = X.notna().all(axis=1)
            X, y = X[mask], y[mask]

            if X.empty:
                print(f"[WARN] Sem dados válidos após limpeza em {csv}")
                continue

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )

            model.fit(X, y)
            y_pred = model.predict(X)

            results.append({
                "camera": cam,
                "week": csv.stem,
                "samples": len(X),
                "accuracy": accuracy_score(y, y_pred),
                "f1": f1_score(y, y_pred)
            })

            joblib.dump(model, model_dir / f"{csv.stem}.joblib")

        if results:
            out = Path("data/results")
            out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(results).to_csv(
                out / f"{cam}_weekly.csv",
                index=False
            )

if __name__ == "__main__":
    train_all()

