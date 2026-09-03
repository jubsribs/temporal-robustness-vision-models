import pandas as pd
from config import SENSORS


def load_sensor(sensor, day_dir):
    path = day_dir / f"{sensor}.csv"

    if not path.exists():
        print(f"[DEBUG] Sensor CSV não encontrado: {path}")
        return None

    if path.stat().st_size == 0:
        print(f"[AVISO] Sensor CSV vazio: {path}")
        return None

    print(f"Lendo arquivo: {path}", flush=True)

    features = SENSORS.get(sensor)

    if not features:
        print(f"[DEBUG] Sensor {sensor} não mapeado")
        return None

    try:
        df = pd.read_csv(path)

    except pd.errors.EmptyDataError:
        print(f"[AVISO] Nenhum dado encontrado no CSV: {path}")
        return None

    except pd.errors.ParserError as error:
        print(f"\n[ERRO] CSV malformado: {path}")
        print(f"[ERRO] {error}")
        raise

    if df.empty:
        print(f"[AVISO] CSV sem registros: {path}")
        return None

    required_columns = ["timestamp"] + features
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"[DEBUG] Colunas ausentes em {path}: {missing_columns}")
        return None

    original_timestamp = df["timestamp"].copy()

    numeric_timestamp = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        numeric_timestamp,
        unit="s",
        utc=True,
        errors="coerce"
    )

    invalid_mask = df["timestamp"].isna()

    if invalid_mask.any():
        print(
            f"[AVISO] {path} possui "
            f"{invalid_mask.sum()} timestamp(s) inválido(s)"
        )
        print(
            original_timestamp.loc[invalid_mask]
            .head(10)
            .to_string(index=False)
        )

        df = df.loc[~invalid_mask].copy()

    if df.empty:
        print(f"[AVISO] Nenhum registro válido restante em: {path}")
        return None

    return df[required_columns]