from config import CAMERAS,PROCESSED_DIR
from build_dataset import build_dataset
from pathlib import Path

def process_all():
    for cam in CAMERAS:
        print(f"Processando {cam}...")
        df = build_dataset(cam)

        df["week"] = df["time_block"].dt.isocalendar().week

        out_dir = Path(PROCESSED_DIR) / cam
        out_dir.mkdir(parents=True, exist_ok=True)

        for week, g in df.groupby("week"):
            g.drop(columns="week").to_csv(
                out_dir / f"week_{week}.csv", index=False
            )

if __name__ == "__main__":
    process_all()
