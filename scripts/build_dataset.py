from config import CAMERAS, SENSORS, BASE_DIR, PROCESSED_DIR
from scripts.load_camera import load_camera
from scripts.load_sensors import load_sensor
from pathlib import Path
import pandas as pd

def build_dataset(camera, week_dirs, out_file):
    merged = []

    for day_dir in week_dirs:
        cam_df = load_camera(camera, day_dir)
        if cam_df is None:
            continue

        dfs = [cam_df]

        for sensor in SENSORS:
            df = load_sensor(sensor, day_dir)
            if df is not None:
                dfs.append(df)

        day_df = dfs[0]
        for df in dfs[1:]:
            day_df = day_df.merge(df, on="timestamp", how="left")

        merged.append(day_df)

    if not merged:
        return None

    return pd.concat(merged).sort_values("timestamp")
