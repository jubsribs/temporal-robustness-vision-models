from pathlib import Path
import pandas as pd

def load_camera(camera_name, base_dir):
    dfs = []

    for day_dir in sorted(Path(base_dir).iterdir()):
        cam_dir = day_dir / camera_name
        csv_path = cam_dir / "ocupacao.csv"

        if not csv_path.exists():
            continue

    
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_block"] = df["timestamp"].dt.floor("10min")

        df = df.groupby("time_block")["ocupada"].max().reset_index()
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"Nenhum dado encontrado para {camera_name}")

    return pd.concat(dfs, ignore_index=True)
