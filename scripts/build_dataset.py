from config import CAMERAS, SENSORS, BASE_DATA_DIR
from load_camera import load_camera
from load_sensors import load_sensor
from pathlib import Path

def build_dataset(camera_name):
    df_final = None

    for sensor in SENSORS:
        df_s = load_sensor(sensor, BASE_DATA_DIR)
        df_final = df_s if df_final is None else df_final.merge(
            df_s, on="time_block", how="outer"
        )

    df_cam = load_camera(camera_name, BASE_DATA_DIR)

    df = df_final.merge(df_cam, on="time_block", how="inner")
    return df.sort_values("time_block")
