from config import CAMERAS,PROCESSED_DIR,BASE_DIR
from scripts.build_dataset import build_dataset
from pathlib import Path
import pandas as pd

def floor_to_10min(df):
    """Arredonda timestamps para janelas de 10 minutos."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.floor("10min")
    return df

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
