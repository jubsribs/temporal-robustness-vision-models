from config import CAMERAS, SENSORS, BASE_DIR, PROCESSED_DIR
from scripts.load_camera import load_camera
from scripts.load_sensors import load_sensor
from pathlib import Path
import pandas as pd

def normalize_timestamp(df):
    if "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Já arredonda aqui
    df["timestamp"] = df["timestamp"].dt.floor("10min")
    return df


def build_dataset(camera, week_dirs, out_file):
    merged = []

    for day_dir in week_dirs:
        cam_df = load_camera(camera, day_dir)
        if cam_df is None:
            continue

        cam_df = normalize_timestamp(cam_df)
        cam_df = cam_df.loc[:, ~cam_df.columns.duplicated()]
        dfs = [cam_df]

        for sensor in SENSORS:
            df = load_sensor(sensor, day_dir)
            if df is not None:
                df = normalize_timestamp(df)
                if "ocupada" in df.columns:
                    df = df.drop(columns=["ocupada"])
                dfs.append(df)

        day_df = dfs[0]
        for df in dfs[1:]:
            day_df = day_df.merge(df, on="timestamp", how="left")

        merged.append(day_df)

    if not merged:
        return None

    return pd.concat(merged).sort_values("timestamp")
