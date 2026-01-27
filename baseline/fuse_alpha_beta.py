from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("data/fused")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAM_ALPHA = "camera_alpha"
CAM_BETA = "camera_beta"


def safe_load_occupancy(csv_path: Path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    if "ocupada" not in df.columns:
        df["ocupada"] = 0

    return df[["timestamp", "ocupada"]]


def fuse_week(week_id: str):
    alpha_csv = PROCESSED_DIR / CAM_ALPHA / f"{week_id}.csv"
    beta_csv = PROCESSED_DIR / CAM_BETA / f"{week_id}.csv"

    if not alpha_csv.exists() or not beta_csv.exists():
        print(f"[SKIP] {week_id}: alpha ou beta ausente")
        return

    df_a = safe_load_occupancy(alpha_csv)
    df_b = safe_load_occupancy(beta_csv)

    df = df_a.merge(
        df_b,
        on="timestamp",
        how="outer",
        suffixes=("_alpha", "_beta")
    ).fillna(0)

    # OR lógico (late fusion)
    df["ocupada"] = (
        (df["ocupada_alpha"] == 1) |
        (df["ocupada_beta"] == 1)
    ).astype(int)

    df_out = df[["timestamp", "ocupada"]].sort_values("timestamp")

    out_file = OUT_DIR / f"{week_id}.csv"
    df_out.to_csv(out_file, index=False)

    print(f"[OK] {week_id}: fusão salva ({len(df_out)} linhas)")


def main():
    alpha_dir = PROCESSED_DIR / CAM_ALPHA

    week_ids = sorted(
        f.stem
        for f in alpha_dir.glob("week_????_??.csv")
    )

    if not week_ids:
        print("[WARN] Nenhuma semana encontrada")
        return

    for week_id in week_ids:
        fuse_week(week_id)


if __name__ == "__main__":
    main()
