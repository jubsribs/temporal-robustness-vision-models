from pathlib import Path
import pandas as pd

def load_sensor(sensor, base_dir):
    dfs = []

    for day in sorted(Path(base_dir).iterdir()):
        path = day / f"{sensor}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["timestamp", "average"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].dt.floor("10min")
            df = df.groupby("timestamp").first().reset_index()
            df = df.rename(columns={"average": sensor})
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None
