from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    dfs = []
    for csv in RESULTS_DIR.glob("*.csv"):
        df = pd.read_csv(csv)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_metric(df, camera, metric):
    cam_df = df[df["camera"] == camera].sort_values("week")

    plt.figure()
    plt.plot(cam_df["week"], cam_df[metric], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel("Semana")
    plt.title(f"{camera} — {metric} por semana")
    plt.tight_layout()

    plt.savefig(OUT_DIR / f"{camera}_{metric}.png")
    plt.close()


if __name__ == "__main__":
    df = load_all_results()

    for cam in df["camera"].unique():
        plot_metric(df, cam, "accuracy")
        plot_metric(df, cam, "f1")
