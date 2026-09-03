import pandas as pd
from config import SENSORS


def load_sensor(sensor, day_dir):
    path = day_dir / f"{sensor}.csv"

    if not path.exists():
        return None

    print(f"Lendo arquivo: {path}", flush=True)

    features = SENSORS.get(sensor)

    if not features:
        print(f"[DEBUG] Sensor {sensor} não mapeado")
        return None

    try:
        # Primeiro carrega o CSV sem converter a data.
        # Assim, problemas estruturais ficam separados dos problemas de timestamp.
        df = pd.read_csv(path)

    except pd.errors.ParserError as error:
        print(f"\n[ERRO] CSV malformado: {path}")
        print(f"[ERRO] {error}")
        print(
            "[ERRO] Verifique a linha indicada no erro. "
            "Ela possui uma quantidade inesperada de campos."
        )
        raise

    if "timestamp" not in df.columns:
        print(f"[DEBUG] Coluna 'timestamp' ausente em {path}")
        return None

    # Converte o timestamp explicitamente.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"[AVISO] {path} possui "
            f"{invalid_timestamps} timestamp(s) inválido(s)"
        )

        # Remove apenas registros sem timestamp válido.
        df = df.dropna(subset=["timestamp"])

    missing = [column for column in features if column not in df.columns]

    if missing:
        print(f"[DEBUG] Colunas ausentes em {path}: {missing}")
        return None

    return df[["timestamp"] + features]