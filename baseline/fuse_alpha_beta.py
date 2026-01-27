from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR

OUT_DIR = Path("data/fused")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "average_humidity",
    "average_gas",
    "average_light",
    "average_loudness",
    "average_temperature",
    "average_object",
    "average_ambient",
    "average_distance",
]

def fuse_week(week_id: str):
    alpha_csv = PROCESSED_DIR / "camera_alpha" / f"{week_id}.csv"
    beta_csv = PROCESSED_DIR / "camera_beta" / f"{week_id}.csv"

    if not alpha_csv.exists() or not beta_csv.exists():
        print(f"[WARN] Semana {week_id}: arquivos ausentes")
        return

    df_a = pd.read_csv(alpha_csv)
    df_b = pd.read_csv(beta_csv)

    # validações mínimas
    for name, df in [("alpha", df_a), ("beta", df_b)]:
        if "timestamp" not in df.columns or "ocupada" not in df.columns:
            raise ValueError(f"{name} sem colunas necessárias")

    # merge por timestamp
    df = df_a.merge(
        df_b,
        on="timestamp",
        how="inner",
        suffixes=("_alpha", "_beta")
    )

    # OR lógico
    df["ocupada_fused"] = (
        (df["ocupada_alpha"] == 1) |
        (df["ocupada_beta"] == 1)
    ).astype(int)

    fused_sensors = {}

    for col in SENSOR_COLS:
        a = f"{col}_alpha"
        b = f"{col}_beta"

        if a in df.columns and b in df.columns:
            fused_sensors[col] = df[a].where(df[a] == df[b])
        else:
            print(f"[INFO] {week_id}: sensor ausente → {col}")

    # Dataset final
    out_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "ocupada": df["ocupada"],
        **fused_sensors
    })


    out_file = OUT_DIR / f"{week_id}_alpha_beta.csv"
    out_df.to_csv(out_file, index=False)

    print(f"[OK] Fusão salva: {out_file}")


def main():
    alpha_dir = PROCESSED_DIR / "camera_alpha"
    weeks = sorted(p.stem for p in alpha_dir.glob("week_*.csv"))

    for week_id in weeks:
        fuse_week(week_id)


if __name__ == "__main__":
    main()
