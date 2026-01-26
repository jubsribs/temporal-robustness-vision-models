from config import CAMERAS,PROCESSED_DIR,BASE_DIR
from scripts.build_dataset import build_dataset
from pathlib import Path
import pandas as pd

def floor_to_10min(df):
    # 1. Validação básica
    if df is None or df.empty:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame sem coluna 'timestamp'")

    # 2. Garantir datetime válido
    df = df.copy()
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("Coluna 'timestamp' não é datetime")

    # 3. Criar bloco temporal
    df["time_block"] = df["timestamp"].dt.floor("10min")

    # 4. Agregação segura
    sensor_cols = [
        c for c in df.columns
        if c not in ["timestamp", "time_block", "ocupada"]
    ]

    agg = {c: "mean" for c in sensor_cols}
    if "ocupada" in df.columns:
        agg["ocupada"] = "max"

    df = (
        df
        .groupby("time_block", as_index=False)
        .agg(agg)
        .rename(columns={"time_block": "timestamp"})
    )

    # 5. Preencher ausências
    return df.fillna(0)

def process_all():

    # Lista todos os dias disponíveis
    days = sorted([d for d in Path(BASE_DIR).iterdir() if d.is_dir()])

    for cam in CAMERAS:
        print(f"Processando {cam}...")
        out_dir = Path(PROCESSED_DIR) / cam
        out_dir.mkdir(parents=True, exist_ok=True)

        weekly_data = {}

        for day_dir in days:
            df = build_dataset(cam, [day_dir], None)

            if df is None or df.empty:
                continue

            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Extrai semana ISO
            iso = df["timestamp"].dt.isocalendar()
            df["iso_year"] = iso.year
            df["iso_week"] = iso.week

            for (year, week), g in df.groupby(["iso_year", "iso_week"]):
                key = f"week_{year}_{week:02d}"
                weekly_data.setdefault(key, []).append(g)

        # Salva semanas corretamente
        for week_id, dfs in weekly_data.items():
            week_df = pd.concat(dfs, ignore_index=True)
             # REMOVE colunas auxiliares de tempo
            week_df = week_df.drop(
                columns=[c for c in ["iso_year", "iso_week"] if c in week_df.columns]
            )

            week_df = floor_to_10min(week_df)

            out_file = out_dir / f"{week_id}.csv"
            week_df.to_csv(out_file, index=False)

            print(f"[INFO] {cam}: {week_id} salvo ({len(week_df)} linhas)")


if __name__ == "__main__":
    process_all()
