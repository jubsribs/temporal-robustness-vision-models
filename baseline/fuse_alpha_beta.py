from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR

OUT_DIR = Path("data/fused")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAM_ALPHA = "camera_alpha"
CAM_BETA = "camera_beta"

def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if "ocupada" not in df.columns:
        df["ocupada"] = 0
    # normaliza ocupada para 0/1
    df["ocupada"] = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)
    return df


def fuse_file(filename: str) -> None:
    alpha_csv = PROCESSED_DIR / CAM_ALPHA / filename
    beta_csv = PROCESSED_DIR / CAM_BETA / filename

    if not alpha_csv.exists() or not beta_csv.exists():
        print(f"[SKIP] {filename}: alpha ou beta ausente")
        return

    df_a = load_df(alpha_csv)
    df_b = load_df(beta_csv)

    # Só precisamos de ocupada por timestamp para calcular a fusão
    occ = df_a[["timestamp", "ocupada"]].rename(columns={"ocupada": "ocupada_alpha"}).merge(
        df_b[["timestamp", "ocupada"]].rename(columns={"ocupada": "ocupada_beta"}),
        on="timestamp",
        how="outer",
    )

    occ["ocupada_alpha"] = occ["ocupada_alpha"].fillna(0).astype(int)
    occ["ocupada_beta"] = occ["ocupada_beta"].fillna(0).astype(int)

    # Lógica pedida (equivalente a OR)
    occ["ocupada"] = ((occ["ocupada_alpha"] == 1) | (occ["ocupada_beta"] == 1)).astype(int)

    # Base final com TODAS as colunas (iguais nas duas câmaras):
    # - juntamos alpha e beta por outer para não perder timestamps
    # - como as colunas são iguais, usamos combine_first para preencher faltas
    df_full = df_a.merge(df_b, on="timestamp", how="outer", suffixes=("_a", "_b"))

    # Reconstruir colunas originais (sem duplicar):
    # Pega todas as colunas de df_a (inclui timestamp) como "schema"
    cols = list(df_a.columns)
    if "timestamp" not in cols:
        cols = ["timestamp"] + cols

    out = pd.DataFrame()
    out["timestamp"] = df_full["timestamp"]

    for c in cols:
        if c == "timestamp":
            continue
        ca = f"{c}_a"
        cb = f"{c}_b"
        # se por algum motivo não existir sufixo (coluna só em um lado), trata também
        if ca in df_full.columns and cb in df_full.columns:
            out[c] = df_full[ca].combine_first(df_full[cb])
        elif c in df_full.columns:
            out[c] = df_full[c]
        elif ca in df_full.columns:
            out[c] = df_full[ca]
        elif cb in df_full.columns:
            out[c] = df_full[cb]
        else:
            out[c] = pd.NA

    # Agora substitui APENAS a ocupada pelo resultado da fusão
    out = out.drop(columns=["ocupada"], errors="ignore").merge(
        occ[["timestamp", "ocupada"]],
        on="timestamp",
        how="left",
    )
    out["ocupada"] = out["ocupada"].fillna(0).astype(int)

    out = out.sort_values("timestamp")

    out_file = OUT_DIR / filename
    out.to_csv(out_file, index=False)
    print(f"[OK] {filename}: fusão salva ({len(out)} linhas)")


def main():
    alpha_dir = PROCESSED_DIR / CAM_ALPHA
    beta_dir = PROCESSED_DIR / CAM_BETA

    alpha_files = {p.name for p in alpha_dir.glob("*.csv")}
    beta_files = {p.name for p in beta_dir.glob("*.csv")}

    # só processa os que existem nos dois lados
    common_files = sorted(alpha_files & beta_files)

    if not common_files:
        print("[WARN] Nenhum CSV comum entre alpha e beta")
        return

    for fname in common_files:
        fuse_file(fname)


if __name__ == "__main__":
    main()
