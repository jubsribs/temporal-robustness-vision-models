from pathlib import Path
import pandas as pd

def load_sensor(sensor, day_dir):
    path = day_dir / f"{sensor}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[["timestamp", "average"]].rename(
        columns={"average": sensor}
    )
