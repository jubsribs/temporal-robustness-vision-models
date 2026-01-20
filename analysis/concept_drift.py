from pathlib import Path
import pandas as pd
from config import RESULTS_DIR

THRESHOLD = 0.10  # 10% de queda


def detect_drift(df):
    df = df.sort_values("week")

    drifts = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        drop = prev["f1"] - curr["f1"]
        if drop >= THRESHOLD:
            drifts.append({
                "camera": curr["camera"],
                "week_prev": prev["week"],
                "week_curr": curr["week"],
                "f1_drop": drop
            })

    return pd.DataFrame(drifts)


if __name__ == "__main__":
    all_results = pd.concat(
        [pd.read_csv(f) for f in RESULTS_DIR.glob("*.csv")],
        ignore_index=True
    )

    drift_df = detect_drift(all_results)

    if drift_df.empty:
        print("Nenhum concept drift detectado.")
    else:
        print("Concept drift detectado:")
        print(drift_df)

        drift_df.to_csv(
            RESULTS_DIR / "concept_drift_report.csv",
            index=False
        )
