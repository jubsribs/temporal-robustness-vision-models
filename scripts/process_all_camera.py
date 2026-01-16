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

    # Agrupa dias em semanas (listas de 7 dias)
    weeks = [days[i:i+7] for i in range(0, len(days), 7)]

    for cam in CAMERAS:
        print(f"Processando {cam}...")
        out_dir = Path(PROCESSED_DIR) / cam
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, week_dirs in enumerate(weeks, start=1):
            # Caminho do CSV de saída
            out_file = out_dir / f"week_{i:02d}.csv"
            
            # Chama build_dataset com semana completa
            df = build_dataset(cam, week_dirs, out_file)

            if df is None or df.empty:
                print(f"[WARN] Nenhum dado para {cam}, semana {i}")
                continue

            # Arredonda timestamps para janelas de 10 minutos
            df = floor_to_10min(df)

            # Salva o CSV final
            df.to_csv(out_file, index=False)
            print(f"[INFO] Semana {i} processada para {cam}")

if __name__ == "__main__":
    process_all()
