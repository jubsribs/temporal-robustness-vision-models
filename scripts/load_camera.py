import pandas as pd


def load_camera(camera, day_dir):
    path = day_dir / camera / "ocupacao.csv"

    if not path.exists():
        print(f"[DEBUG] Camera CSV não encontrado: {path}")
        return None

    print(f"Lendo câmera: {path}", flush=True)

    try:
        df = pd.read_csv(path)

    except pd.errors.ParserError as error:
        print(f"[ERRO] CSV da câmera malformado: {path}")
        print(f"[ERRO] {error}")
        raise

    required_columns = ["timestamp", "ocupada"]
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
        print("[DEBUG] Exemplos:")
        print(
            original_timestamp.loc[invalid_mask]
            .head(10)
            .to_string(index=False)
        )

        df = df.loc[~invalid_mask].copy()

    return df[required_columns]