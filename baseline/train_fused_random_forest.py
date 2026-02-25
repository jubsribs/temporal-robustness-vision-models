import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle
import os

from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    classification_report,
)
from scipy.stats import ttest_ind

# Diretórios
FUSED_DIR = Path("data/fused")
BASE_DIR = Path("analysis/random_forest_fused")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = BASE_DIR / timestamp

FIG_DIR = RUN_DIR / "figures"
METRICS_DIR = RUN_DIR / "metrics"
MODEL_DIR = RUN_DIR / "model"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_fused_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)

    if "ocupada" not in df.columns:
        raise ValueError(f"{csv_path.name} não tem coluna 'ocupada'")

    df["ocupada"] = pd.to_numeric(df["ocupada"], errors="coerce").fillna(0).astype(int)
    return df


def _is_bad_metric(x: float) -> bool:
    #  NaN (indefinição/denominador zero)
    if x is None:
        return True
    try:
        if np.isnan(x):
            return True
    except Exception:
        pass
    return x == 0


def _value_or_nan(x: float) -> float:
    # Para barras:  mete NaN para a barra não aparecer
    return np.nan if _is_bad_metric(x) else float(x)

def analyze_occupancy_episodes(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    episodes = []
    in_episode = False
    start_idx = None

    # -----------------------------
    # 1. Detectar blocos contínuos
    # -----------------------------
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_episode:
            in_episode = True
            start_idx = i

        elif y_true[i] == 0 and in_episode:
            end_idx = i - 1
            episodes.append((start_idx, end_idx))
            in_episode = False

    # Caso termine em episódio
    if in_episode:
        episodes.append((start_idx, len(y_true) - 1))

    # -----------------------------
    # 2. Analisar cada episódio
    # -----------------------------
    results = []

    for (start, end) in episodes:
        true_segment = y_true[start:end+1]
        pred_segment = y_pred[start:end+1]

        duration = end - start + 1
        detected_positions = np.where(pred_segment == 1)[0]

        if len(detected_positions) == 0:
            status = "missed"
            delay = None
            coverage = 0.0

        else:
            first_detection = detected_positions[0]
            delay = first_detection  # slots após início real
            coverage = len(detected_positions) / duration

            if delay == 0 and coverage == 1.0:
                status = "fully_detected"
            elif delay > 0:
                status = "delayed"
            else:
                status = "partial"

        results.append({
            "start_index": start,
            "end_index": end,
            "duration_slots": duration,
            "detected": len(detected_positions) > 0,
            "delay_slots": delay,
            "coverage_ratio": coverage,
            "status": status
        })

    df_results = pd.DataFrame(results)

    # -----------------------------
    # 3. Métrica global episódio
    # -----------------------------
    if len(df_results) > 0:
        episode_detection_rate = df_results["detected"].mean()
    else:
        episode_detection_rate = np.nan

    print(f"\nTotal episódios reais: {len(df_results)}")
    print(f"OEDR (episode detection rate): {episode_detection_rate:.3f}")

    return df_results

def train_fused_random_forest(random_state: int = 42, threshold: float = 0.35):
    csvs = sorted(FUSED_DIR.glob("week_*.csv"))
    if not csvs:
        print("[WARN] Nenhum CSV found em data/fused")
        return

    dfs = []
    for p in csvs:
        try:
            dfs.append(_load_fused_csv(p))
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")

    if not dfs:
        print("[WARN] Nenhum CSV fused válido para treino.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Features (numéricas) + target
    X = (
        df.drop(columns=["timestamp", "ocupada"], errors="ignore")
        .select_dtypes(include=["number"])
        .fillna(0)
    )
    print(f"[INFO] Features usadas: {list(X.columns)}")

    y = df["ocupada"]

    if X.empty or y.empty:
        print("[WARN] Dataset vazio após pré-processamento.")
        return

    # split 80/20 global
    strat = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=strat,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    # =========================
    # TEMPO + MEMÓRIA TREINO
    # =========================
    tracemalloc.start()
    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start_train
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # =========================
    # TEMPO PREDIÇÃO
    # =========================
    start_pred = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)
    predict_time = time.perf_counter() - start_pred

    test_df = X_test.copy()
    test_df["y_true"] = y_test.values
    test_df["y_pred"] = y_pred
    test_df["timestamp"] = df.loc[X_test.index, "timestamp"]

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    test_df["episode_status"] = "none"

    episode_df = analyze_occupancy_episodes(
        test_df["y_true"].values,
        test_df["y_pred"].values
    )

    for _, row in episode_df.iterrows():
        start = int(row["start_index"])
        end = int(row["end_index"])
        status = row["status"]

        test_df.loc[start:end, "episode_status"] = status

    missed_df = df_test[df_test["episode_status"] == "missed"]
    detected_df = df_test[df_test["episode_status"] == "fully_detected"]

    print("Missed samples:", len(missed_df))
    print("Detected samples:", len(detected_df))

    episode_df.to_csv(METRICS_DIR / "episode_analysis.csv", index=False)
    #---------------------------------------
    # Teste de Welch
    # -----------------------------------------

    results = []

    feature_cols = X_test.columns

    for col in feature_cols:
        missed_vals = missed_df[col].dropna()
        detected_vals = detected_df[col].dropna()

    if len(missed_vals) > 1 and len(detected_vals) > 1:
        stat, pval = ttest_ind(missed_vals, detected_vals, equal_var=False)
        
        results.append({
            "feature": col,
            "missed_mean": missed_vals.mean(),
            "detected_mean": detected_vals.mean(),
            "difference": missed_vals.mean() - detected_vals.mean(),
            "p_value": pval
        })

    stats_df = pd.DataFrame(results).sort_values("p_value")

    print(stats_df.head(10))

    top_features = stats_df.head(5)["feature"]

    for col in top_features:
        plt.figure()
        plt.hist(missed_df[col], alpha=0.5, label="Missed")
        plt.hist(detected_df[col], alpha=0.5, label="Detected")
        plt.legend()
        plt.title(col)
        plt.show()

    # --------------------------------------------------
    # Análise de erros (Falsos Negativos)
    # --------------------------------------------------
    results_df = X_test.copy()
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = y_pred
    results_df["y_prob"] = y_prob

    false_negatives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 0)
    ]

    true_positives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 1)
    ]

    if len(false_negatives) > 1 and len(true_positives) > 1:

        comparison = pd.DataFrame({
            "FN_mean": false_negatives.mean(numeric_only=True),
            "TP_mean": true_positives.mean(numeric_only=True),
        })

        comparison["difference"] = (
            comparison["FN_mean"] - comparison["TP_mean"]
        )

        comparison = comparison.sort_values(
            "difference",
            key=lambda x: np.abs(x),
            ascending=False
        )

        comparison.to_csv(
            METRICS_DIR / "false_negative_analysis.csv"
        )

        print(f"[INFO] FN analysis saved ({len(false_negatives)} FN samples)")
    else:
        print("[INFO] Not enough FN/TP samples for statistical comparison")
    
    false_negatives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 0)
    ]

    true_positives = results_df[
        (results_df["y_true"] == 1) &
        (results_df["y_pred"] == 1)
    ]

    if len(false_negatives) > 0:
        print(
            "[INFO] Mean probability (FN):",
            false_negatives["y_prob"].mean()
        )

    if len(true_positives) > 0:
        print(
            "[INFO] Mean probability (TP):",
            true_positives["y_prob"].mean()
    )
    comparison.sort_values(
        "difference",
        key=lambda x: np.abs(x),
        ascending=False
    ).to_csv(
        METRICS_DIR / "false_negative_analysis.csv"
    )
    # =========================
    # TAMANHO DO MODELO
    # =========================
    model_path = MODEL_DIR / "random_forest_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(
        f"[PERFORMANCE] "
        f"train_time={train_time:.4f}s | "
        f"predict_time={predict_time:.4f}s | "
        f"peak_memory={peak_memory / (1024*1024):.2f}MB | "
        f"model_size={model_size_mb:.2f}MB"
    )


    # Métricas globais
    acc = accuracy_score(y_test, y_pred)

    # Métricas por classe (1 = Ocupado, 0 = Ausência/Livre)
    # zero_division=np.nan para capturar indefinições
    prec_occ = precision_score(y_test, y_pred, pos_label=1, zero_division=np.nan)
    rec_occ = recall_score(y_test, y_pred, pos_label=1, zero_division=np.nan)
    f1_occ = f1_score(y_test, y_pred, pos_label=1, zero_division=np.nan)

    prec_abs = precision_score(y_test, y_pred, pos_label=0, zero_division=np.nan)
    rec_abs = recall_score(y_test, y_pred, pos_label=0, zero_division=np.nan)
    f1_abs = f1_score(y_test, y_pred, pos_label=0, zero_division=np.nan)

    print(
        f"[FUSED RF 80/20] samples={len(df)} | "
        f"train={len(X_train)} | test={len(X_test)} | acc={acc:.3f}"
    )

    # Report (para suportes)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Absence(0)", "Occupied(1)"],
        zero_division=np.nan,
        output_dict=True,
    )

    metrics_raw = {
        "split": "80/20",
        "random_state": random_state,
        "samples_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "peak_memory_MB": peak_memory / (1024 * 1024),
        "model_size_MB": model_size_mb,
        "accuracy": acc,
        "precision_occupied": prec_occ,
        "recall_occupied": rec_occ,
        "f1_occupied": f1_occ,
        "precision_absence": prec_abs,
        "recall_absence": rec_abs,
        "f1_absence": f1_abs,
        "threshold": threshold,
        "support_absence": int(report["Absence(0)"]["support"]) if "Absence(0)" in report else None,
        "support_occupied": int(report["Occupied(1)"]["support"]) if "Occupied(1)" in report else None,
    }

    # ✅ CSV apenas com números (sem “◯”)
    pd.DataFrame([metrics_raw]).to_csv(
        METRICS_DIR / "fused_random_forest_metrics.csv",
        index=False,
    )
    print("[OK] Métricas salvas em", METRICS_DIR)

    # =========================
    # GRÁFICO DE MÉTRICAS (com “círculo oco” quando 0/NaN)
    # =========================
    plot_vals_raw = {
        "accuracy": acc,
        "prec_occupied": prec_occ,
        "rec_occupied": rec_occ,
        "f1_occupied": f1_occ,
        "prec_absence": prec_abs,
        "rec_absence": rec_abs,
        "f1_absence": f1_abs,
    }

    labels = list(plot_vals_raw.keys())
    raw_values = [plot_vals_raw[k] for k in labels]
    values = [_value_or_nan(v) for v in raw_values]  # NaN onde for mau

    plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))

    plt.bar(x, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Random Forest — Fused (80/20)")
    plt.xticks(x, labels, rotation=30, ha="right")

    # desenha círculo oco por cima do label quando a métrica for 0/NaN
    for i, v in enumerate(raw_values):
        if _is_bad_metric(v):
            plt.scatter(
                [i],
                [0.05],            # altura fixa para ficar visível (ajusta se quiseres)
                marker="o",
                facecolors="none", # círculo oco
            )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_metrics_bar.png")
    plt.close()

    # =========================
    # MATRIZ DE CONFUSÃO
    # =========================
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0,1],
        display_labels=["Absence(0)", "Occupied(1)"],
        cmap="Blues",
        normalize=None,
    )
    disp.ax_.set_title("Confusion Matrix — Fused RF (80/20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fused_confusion_matrix.png")
    plt.close()

    print("[OK] Gráficos salvos", FIG_DIR)



if __name__ == "__main__":
    train_fused_random_forest()
