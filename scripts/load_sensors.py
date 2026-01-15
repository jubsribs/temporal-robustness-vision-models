from pathlib import Path
import pandas as pd
from config import SENSORS

def load_sensor(sensor, day_dir):
    path = day_dir / f"{sensor}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"],date_format="%Y-%m-%d %H:%M:%S")


    features = SENSORS.get(sensor)
    if not features:
        print(f"[DEBUG] Sensor {sensor} não mapeado")
        return None

    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[DEBUG] Colunas ausentes em {path}: {missing}")
        return None

    return df[["timestamp"] + features]
