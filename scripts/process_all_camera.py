from config import CAMERAS, PROCESSED_DIR, BASE_DIR
from scripts.build_dataset import build_dataset
from pathlib import Path
import pandas as pd


def aggregate_time_blocks(df):
    """
    Garante uma única linha por bloco de 10 minutos.

    - Sensores: média
    - Ocupação: máximo
    - Valores ausentes: permanecem como NaN
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame sem coluna 'timestamp'")

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].dt.floor("10min")

    aggregate_functions = {}

    for column in df.columns:
        if column == "timestamp":
            continue

        if column == "ocupada":
            aggregate_functions[column] = "max"
        elif pd.api.types.is_numeric_dtype(df[column]):
            aggregate_functions[column] = "mean"

    if not aggregate_functions:
        return df[["timestamp"]].drop_duplicates()

    return (
        df.groupby("timestamp", as_index=False)
        .agg(aggregate_functions)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def process_all():
    base_dir = Path(BASE_DIR)
    processed_dir = Path(PROCESSED_DIR)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Diretório de dados não encontrado: {base_dir}"
        )

    days = sorted(
        directory
        for directory in base_dir.iterdir()
        if directory.is_dir()
    )

    if not days:
        print(f"[AVISO] Nenhum diretório diário encontrado em {base_dir}")
        return

    for camera in CAMERAS:
        print(f"\nProcessando {camera}...")

        out_dir = processed_dir / camera
        out_dir.mkdir(parents=True, exist_ok=True)

        weekly_data = {}

        for day_dir in days:
            print(f"[INFO] Dia: {day_dir.name}")

            df = build_dataset(
                camera,
                [day_dir],
                None
            )

            if df is None or df.empty:
                print(
                    f"[AVISO] Nenhum dado válido para "
                    f"{camera} em {day_dir.name}"
                )
                continue

            df = df.copy()

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce"
            )

            df = df.dropna(subset=["timestamp"])

            if df.empty:
                continue

            iso_calendar = df["timestamp"].dt.isocalendar()

            df["iso_year"] = iso_calendar["year"].astype(int)
            df["iso_week"] = iso_calendar["week"].astype(int)

            for (year, week), group in df.groupby(
                ["iso_year", "iso_week"]
            ):
                week_id = f"week_{int(year)}_{int(week):02d}"

                weekly_data.setdefault(
                    week_id,
                    []
                ).append(group.copy())

        for week_id, dataframes in sorted(weekly_data.items()):
            week_df = pd.concat(
                dataframes,
                ignore_index=True,
                sort=False
            )

            week_df = week_df.drop(
                columns=["iso_year", "iso_week"],
                errors="ignore"
            )

            week_df = aggregate_time_blocks(week_df)

            if week_df.empty:
                print(
                    f"[AVISO] {camera}: {week_id} "
                    "não possui registros válidos"
                )
                continue

            out_file = out_dir / f"{week_id}.csv"

            week_df.to_csv(
                out_file,
                index=False
            )

            print(
                f"[INFO] {camera}: {week_id} salvo "
                f"({len(week_df)} linhas) em {out_file}"
            )


if __name__ == "__main__":
    process_all()