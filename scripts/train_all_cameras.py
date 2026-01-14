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
            df = pd.read_csv(csv)

            X = df.drop(columns=["ocupada", "time_block"])
            y = df["ocupada"]

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )

            model.fit(X, y)
            y_pred = model.predict(X)

            results.append({
                "camera": cam,
                "week": csv.stem,
                "accuracy": accuracy_score(y, y_pred),
                "f1": f1_score(y, y_pred)
            })

            joblib.dump(model, model_dir / f"{csv.stem}.joblib")

        pd.DataFrame(results).to_csv(
            f"results/{cam}_weekly.csv", index=False
        )

if __name__ == "__main__":
    train_all()
