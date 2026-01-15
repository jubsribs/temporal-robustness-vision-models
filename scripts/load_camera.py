from pathlib import Path
import pandas as pd

def load_camera(camera, day_dir):
    path = day_dir / camera / f"ocupacao.csv"

    if not path.exists():
        print(f"[DEBUG] Camera CSV não encontrado: {path}")
        return None

    return pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )[["timestamp", "ocupada"]]
