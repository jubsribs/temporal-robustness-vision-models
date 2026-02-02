from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import PROCESSED_DIR, RESULTS_DIR

CAMERAS = ["camera_alpha", "camera_beta"]

# ---- parâmetros da janela deslizante ----
WINDOW = "7D"         # tamanho da janela de treino
STEP = "10min"        # frequência a que executas (10min, 1H, 1D, ...)
TEST_HORIZON = "10min"  # quanto tempo “à frente” usas para avaliar

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = (
        df.drop(columns=["ocupada", "timestamp"], errors="ignore")
          .select_dtypes(include=["number"])
          .fillna(0)
    )
    X.columns = X.columns.astype(str)
    return X


def metric_pack(y_true, y_pred):
    # zero_division=np.nan => indefinido vira NaN
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_occ": precision_score(y_true, y_pred, pos_label=1, zero_division=np.nan),
        "recall_occ": recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan),
        "f1_occ": f1_score(y_true, y_pred, pos_label=1, zero_division=np.nan),
        "precision_abs": precision_score(y_true, y_pred, pos_label=0, zero_division=np.nan),
        "recall_abs": recall_score(y_true, y_pred, pos_label=0, zero_division=np.nan),
        "f1_abs": f1_score(y_true, y_pred, pos_label=0, zero_division=np.nan),
    }


def get_feature_space(df: pd.DataFrame) -> list[str]:
    # união de features numéricas no dataset todo (por câmara)
    X = build_X(df)
    return sorted(X.columns.tolist())


def align_X(X: pd.DataFrame, feature_space: list[str]) -> pd.DataFrame:
    return X.reindex(columns=feature_space, fill_value=0)


def load_all_csvs_for_camera(camera: str) -> pd.DataFrame:
    data_dir = PROCESSED_DIR / camera
    csvs = sorted(data_dir.glob("week_*.csv"))
    if not csvs:
        return pd.DataFrame()

    parts = []
    for csv in csvs:
        df = pd.read_csv(csv, parse_dates=["timestamp"])
        if "ocupada" not in df.columns:
            continue
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    df_all = pd.concat(parts, ignore_index=True)

    # ordena por tempo e limpa timestamps inválidos
    df_all = df_all.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # target para int (0/1)
    df_all["ocupada"] = pd.to_numeric(df_all["ocupada"], errors="coerce").fillna(0).astype(int)

    return df_all


def sliding_window_train_eval(camera: str) -> pd.DataFrame:
    df = load_all_csvs_for_camera(camera)
    if df.empty:
        print(f"[WARN] {camera}: sem dados")
        return pd.DataFrame()

    feature_space = get_feature_space(df)

    # cria uma grelha temporal (t) para correr o pipeline
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    # para haver janela + horizonte de teste
    window_td = pd.Timedelta(WINDOW)
    test_td = pd.Timedelta(TEST_HORIZON)

    # começa quando já existe pelo menos WINDOW para trás
    start_t = t_min + window_td
    end_t = t_max - test_td

    if start_t >= end_t:
        print(f"[WARN] {camera}: intervalo insuficiente para WINDOW={WINDOW} e TEST_HORIZON={TEST_HORIZON}")
        return pd.DataFrame()

    grid = pd.date_range(start=start_t, end=end_t, freq=STEP)

    rows = []
    for t in grid:
        train_start = t - window_td
        train_end = t

        test_start = t
        test_end = t + test_td

        train_df = df[(df["timestamp"] > train_start) & (df["timestamp"] <= train_end)]
        test_df = df[(df["timestamp"] > test_start) & (df["timestamp"] <= test_end)]

        # se não há dados suficientes, salta
        if train_df.empty or test_df.empty:
            continue

        X_train = align_X(build_X(train_df), feature_space)
        y_train = train_df["ocupada"].to_numpy()

        X_test = align_X(build_X(test_df), feature_space)
        y_test = test_df["ocupada"].to_numpy()

        # se treino só tem uma classe, SGD pode treinar mas fica degenerado
        if np.unique(y_train).size < 2:
            continue

        model = SGDClassifier(loss="log_loss", random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        m = metric_pack(y_test, y_pred)
        rows.append({
            "camera": camera,
            "t": t,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            **m,
        })

    out = pd.DataFrame(rows)
    return out


if __name__ == "__main__":
    all_out = []
    for cam in CAMERAS:
        out = sliding_window_train_eval(cam)
        if not out.empty:
            all_out.append(out)

    if not all_out:
        print("[WARN] Nenhum resultado gerado")
    else:
        df_res = pd.concat(all_out, ignore_index=True)
        out_csv = RESULTS_DIR / f"sliding_window_{WINDOW}_{STEP}_test_{TEST_HORIZON}.csv"
        df_res.to_csv(out_csv, index=False)
        print(f"[OK] Guardado: {out_csv} | linhas={len(df_res)}")
