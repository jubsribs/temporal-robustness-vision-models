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
      
        cumulative_X = []
        cumulative_y = []

        results = []
    

        for csv in sorted(data_dir.glob("week_*.csv")):
            print(f"  → Semana {csv.stem}")

            df = pd.read_csv(csv, parse_dates=["timestamp"])

            if df.empty or "ocupada" not in df.columns:
                print(f"[WARN] CSV inválido: {csv}")
                continue

            X = df.drop(columns=["ocupada", "timestamp"])
            y = df["ocupada"]

            # garante consistência
            X = X.fillna(0)

            # acumula dados
            cumulative_X.append(X)
            cumulative_y.append(y)

            X_train = pd.concat(cumulative_X, ignore_index=True)
            y_train = pd.concat(cumulative_y, ignore_index=True)

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, zero_division=0)

            results.append({
                "camera": cam,
                "week": csv.stem,
                "train_samples": len(X_train),
                "test_samples": len(X),
                "accuracy": acc,
                "f1": f1
            })

            joblib.dump(model, model_dir / f"{csv.stem}_cumulative.joblib")

            print(f" {csv.stem}: train={len(X_train)}, test={len(X)}, acc={acc:.3f}, f1={f1:.3f}")

        if results:
            out = Path("data/results")
            out.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(results).to_csv(
                out / f"{cam}_weekly_cumulative.csv",
                index=False
            )
if __name__ == "__main__":
    train_all()

