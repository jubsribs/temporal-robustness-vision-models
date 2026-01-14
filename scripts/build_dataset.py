from config import CAMERAS, SENSORS, BASE_DATA_DIR, PROCESSED_DIR
from load_camera import load_camera
from load_sensors import load_sensor
from pathlib import Path

def build_dataset(camera, week_dirs, out_dir):
    merged = None

    for day in week_dirs:
        cam_df = load_camera(camera, day)
        if cam_df is None:
            continue

        sensor_dfs = [cam_df]

        for s in SENSORS:
            df = load_sensor(s, day)
            if df is not None:
                sensor_dfs.append(df)

        day_df = sensor_dfs[0]
        for df in sensor_dfs[1:]:
            day_df = day_df.merge(df, on="timestamp", how="inner")

        merged = day_df if merged is None else pd.concat([merged, day_df])

    if merged is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir, index=False)

def main():
    days = sorted([d for d in BASE_DATA_DIR.iterdir() if d.is_dir()])

    weeks = [days[i:i+7] for i in range(0, len(days), 7)]

    for camera in CAMERAS:
        cam_dir = PROCESSED_DIR / camera

        for i, week in enumerate(weeks, start=1):
            out = cam_dir / f"week_{i:02d}.csv"
            build_dataset(camera, week, out)

if __name__ == "__main__":
    main()
