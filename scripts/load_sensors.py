import csv
import pandas as pd

from config import SENSORS


def read_malformed_csv(path):
    """
    Lê um CSV com quantidade irregular de campos.

    - Campos excedentes no final são descartados.
    - Campos ausentes são preenchidos com None.
    - Cada ocorrência é registrada no terminal.
    """
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file)

        header = next(reader, None)

        if not header:
            return None

        expected_fields = len(header)

        for line_number, row in enumerate(reader, start=2):
            # Ignora linhas completamente vazias.
            if not row or not any(value.strip() for value in row):
                continue

            received_fields = len(row)

            if received_fields > expected_fields:
                print(
                    f"[CORREÇÃO] {path}, linha {line_number}: "
                    f"esperados {expected_fields} campos, "
                    f"encontrados {received_fields}. "
                    "Campos excedentes removidos."
                )

                row = row[:expected_fields]

            elif received_fields < expected_fields:
                print(
                    f"[CORREÇÃO] {path}, linha {line_number}: "
                    f"esperados {expected_fields} campos, "
                    f"encontrados {received_fields}. "
                    "Campos ausentes preenchidos com NaN."
                )

                row.extend(
                    [None] * (expected_fields - received_fields)
                )

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=header)

    return pd.DataFrame(rows, columns=header)


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
        print(f"[AVISO] CSV malformado: {path}")
        print(f"[AVISO] {error}")
        print("[AVISO] Aplicando leitura tolerante.")

        df = read_malformed_csv(path)

    if df is None or df.empty:
        print(f"[AVISO] CSV sem registros: {path}")
        return None

    required_columns = ["timestamp"] + features

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            f"[DEBUG] Colunas ausentes em {path}: "
            f"{missing_columns}"
        )
        return None

    # Converte os campos dos sensores novamente para números,
    # pois a leitura tolerante inicialmente produz strings.
    for column in features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

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

    invalid_timestamp = df["timestamp"].isna()

    if invalid_timestamp.any():
        print(
            f"[AVISO] {path} possui "
            f"{invalid_timestamp.sum()} timestamp(s) inválido(s)"
        )
        print(
            original_timestamp.loc[invalid_timestamp]
            .head(10)
            .to_string(index=False)
        )

        df = df.loc[~invalid_timestamp].copy()

    if df.empty:
        print(f"[AVISO] Nenhum registro válido restante em {path}")
        return None

    return df[required_columns]