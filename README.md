# Temporal Robustness of Multimodal Sensor Fusion for Edge-Based Occupancy Detection

This repository contains the data-processing, training, and evaluation pipeline used to investigate the temporal robustness of edge-based indoor occupancy detection. The framework combines heterogeneous environmental sensing with camera-derived occupancy labels and compares Random Forest (RF) and Stochastic Gradient Descent (SGD) classifiers using both sample-level and episode-level metrics.

The experiments were conducted in a real smart-classroom deployment using a Raspberry Pi 4 Model B. Camera images were used to generate ground-truth occupancy labels, while the deployed classifiers used only non-visual environmental features as inputs.

## Research objectives

- Integrate asynchronous environmental sensor measurements and camera-derived occupancy labels.
- Evaluate supervised occupancy classifiers under edge-computing constraints.
- Compare predictive performance, temporal reliability, memory consumption, training time, inference time, and model size.
- Evaluate occupancy as a temporal process rather than as independent samples.
- Analyze the trade-offs between accuracy, temporal robustness, computational cost, deployment complexity, and privacy.

## Repository structure

```text
.
├── analysis/                 # Metrics, figures, trained models, and analysis scripts
│   ├── random_forest_fused/  # Random Forest experiment outputs
│   └── sgd_fused/            # SGD experiment outputs
├── baseline/                 # Training, fusion, and temporal-baseline scripts
│   ├── fuse_alpha_beta.py
│   ├── train_fused_random_forest.py
│   ├── train_fused_sgd.py
│   ├── train_rf_sliding_window.py
│   └── train_sgd_sliding_window.py
├── data/
│   ├── fused/                # Weekly datasets used by the fused experiments
│   ├── processed/            # Weekly datasets for individual cameras
│   └── results/              # Consolidated evaluation results
├── scripts/                  # Sensor loading, preprocessing, and dataset construction
├── config.py                 # Sensor, camera, and sampling configuration
├── requirements.txt          # Python dependencies
└── README.md
```
## Reproducibility snapshot

The results reported in the IEEE ISC2 2026 paper correspond to commit [`2bc55e3`](https://github.com/jubsribs/temporal-robustness-vision-models/tree/2bc55e3). The `main` branch contains data collected after the experiments reported in the paper and may therefore produce different sample counts and results.

To retrieve the paper snapshot:

```bash
git clone https://github.com/jubsribs/temporal-robustness-vision-models.git
cd temporal-robustness-vision-models
git checkout 2bc55e3
```

## Dataset

The paper snapshot contains 3,382 samples collected over 43 days, from December 16, 2025, to February 23, 2026. The samples are stored in nine weekly CSV files under `data/fused/`.

### Class distribution

| Class | Samples | Percentage |
|---|---:|---:|
| Unoccupied (`0`) | 3,214 | 95.03% |
| Occupied (`1`) | 168 | 4.97% |
| **Total** | **3,382** | **100%** |

The dataset is strongly imbalanced because data collection began near the end of the academic term, when the classroom was unoccupied for long periods.

### Input features

Each fused sample contains eight environmental features:

| Feature | Description |
|---|---|
| `average_gas` | Average MQ-2 gas-sensor reading |
| `average_light` | Average visible-light measurement |
| `average_loudness` | Average sound-level measurement |
| `average_temperature` | Average ambient temperature from the DHT11 sensor |
| `average_humidity` | Average relative humidity from the DHT11 sensor |
| `average_object` | Average object temperature from the infrared thermal sensor |
| `average_ambient` | Average ambient temperature from the infrared thermal sensor |
| `average_distance` | Average distance measured by the ultrasonic sensor |

The binary target column is `ocupada`, where `1` represents occupied and `0` represents unoccupied. The `timestamp` column identifies the associated 10-minute interval.

## Ground-truth generation

Images from the Alpha and Beta cameras were processed with YOLOv8n. Only detections assigned to the `person` class were considered. A camera label was set to `1` when at least one person was detected and to `0` otherwise.

The two camera labels were fused using a logical OR rule:

```text
occupied = occupied_alpha OR occupied_beta
```

Thus, an interval was labeled as occupied when either camera detected at least one person. The fusion implementation is available in `baseline/fuse_alpha_beta.py`.

The images were used for ground-truth generation; RF and SGD inference used only the eight environmental features. This design reduces dependence on continuous image-based occupancy inference, although the presence of cameras means that privacy risks are reduced rather than eliminated.

### YOLO configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8n (`yolov8n.pt`) |
| Target class | Person (`class_id=0`) |
| Occupancy rule | At least one detected person |
| Confidence threshold | **0.25, if the Ultralytics default was used; confirm before publication** |

Raw classroom images are not distributed because they may contain personally identifiable visual information. The repository provides the derived occupancy labels required to reproduce the classifier experiments.

## Preprocessing pipeline

The preprocessing procedure is implemented in `scripts/build_dataset.py`, `scripts/load_camera.py`, `scripts/load_sensors.py`, and `baseline/fuse_alpha_beta.py`.

1. Unix timestamps expressed in seconds are converted to datetimes.
2. Invalid timestamps are discarded.
3. Sensor values are converted to numeric values; malformed or empty sensor files are handled explicitly.
4. Timestamps are floored to fixed 10-minute intervals.
5. Camera labels and sensor measurements are joined by timestamp.
6. Weekly files are generated for each camera.
7. Alpha and Beta camera labels are combined with the logical OR rule.
8. During training, missing numeric feature values are replaced with zero.
9. The `timestamp` and `ocupada` columns are excluded from the model inputs.
10. No feature scaling or standardization is applied in the reported RF and SGD experiments.

### Rationale for the 10-minute interval

The 10-minute interval matches the sensor acquisition schedule and provides a common temporal reference for aligning asynchronous camera and environmental measurements. It also reduces short-term sensor noise, storage requirements, and edge-processing overhead. This choice prioritizes the detection of sustained occupancy episodes; consequently, short visits occurring entirely within or between intervals may not be represented adequately.

## Experimental protocol

The baseline results reported in the paper use a stratified random 80/20 split:

| Parameter | Value |
|---|---|
| Total samples | 3,382 |
| Training samples | 2,705 |
| Test samples | 677 |
| Test proportion | 0.20 |
| Stratification | Occupancy class |
| Random state | 42 |
| Decision threshold | 0.35 |

Because randomly splitting temporally correlated data may introduce temporal leakage, the repository also contains sliding-window scripts for temporally ordered evaluation. Results obtained from the random split and temporal evaluation should be reported separately.

## Model configurations

### Random Forest

The fused Random Forest experiment is implemented in `baseline/train_fused_random_forest.py`.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `criterion` | `gini` |
| `max_depth` | `None` |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | `sqrt` |
| `bootstrap` | `True` |
| `class_weight` | `balanced` |
| `n_jobs` | -1 |
| `random_state` | 42 |
| Probability threshold | 0.35 |

### SGDClassifier

The fused SGD experiment is implemented in `baseline/train_fused_sgd.py`.

| Hyperparameter | Value |
|---|---|
| `loss` | `log_loss` |
| `penalty` | `l2` |
| `alpha` | 0.0001 |
| `max_iter` | 5,000 |
| `tol` | 0.001 |
| `learning_rate` | `optimal` |
| `fit_intercept` | `True` |
| `shuffle` | `True` |
| `early_stopping` | `False` |
| `class_weight` | `balanced` |
| `random_state` | 42 |
| Probability threshold | 0.35 |

Parameters not explicitly passed to the constructors use the defaults of the scikit-learn version recorded for the experiment.

## Evaluation metrics

The following sample-level metrics are reported:

- Accuracy
- Precision, recall, and F1-score for the occupied class
- Precision, recall, and F1-score for the unoccupied class
- Confusion matrix
- Training time
- Inference time
- Peak Python memory allocation measured with `tracemalloc`
- Serialized model size

### Occupancy Episode Detection Rate

An occupancy episode is defined as a maximal sequence of consecutive test samples for which the ground-truth label is occupied. An episode is considered detected when at least one sample within the episode is predicted as occupied.

```text
OEDR = number of detected occupancy episodes / total number of occupancy episodes
```

The episode analysis also reports:

- Episode duration in 10-minute slots
- Delay until the first positive detection
- Coverage ratio within each episode
- Episode status: `fully_detected`, `delayed`, `partial`, or `missed`

## Results reproduced from the paper snapshot

| Metric | Random Forest | SGDClassifier |
|---|---:|---:|
| Accuracy | 0.9764 | 0.9734 |
| Occupied-class precision | 0.7647 | 0.7857 |
| Occupied-class recall | 0.7647 | 0.6471 |
| Occupied-class F1-score | 0.7647 | 0.7097 |
| OEDR | 92.9% (13/14) | 71.4% (10/14) |
| Training time | 8.516 s | 0.063 s |
| Inference time | 0.239 s | 0.0031 s |
| Peak memory allocation | 1.363 MB | 0.281 MB |
| Serialized model size | 2.896 MB | 0.00117 MB |

Runtime measurements depend on the hardware, operating-system load, Python version, and library versions. Small variations are therefore expected when the experiments are repeated.

The corresponding output files are located at:

```text
analysis/random_forest_fused/2026-02-25_12-23-45/
analysis/sgd_fused/2026-02-25_14-55-13/
```

## Hardware and software environment

The edge platform used in the study was a Raspberry Pi 4 Model B.

| Component | Specification |
|---|---|
| Device | Raspberry Pi 4 Model B |
| Processor | Broadcom BCM2711, quad-core Arm Cortex-A72 |
| RAM | **Confirm installed capacity before publication** |
| Operating system | **Confirm distribution and version before publication** |
| Python | Python 3.10 or later; record the exact experiment version |
| pandas | 2.0 or later; record the exact experiment version |
| NumPy | 1.24 or later; record the exact experiment version |
| scikit-learn | 1.3 or later; record the exact experiment version |

Use the following commands on the experiment device to record the missing environment information:

```bash
cat /proc/cpuinfo
free -h
cat /etc/os-release
python3 --version
python3 -m pip freeze
```

For strict reproducibility, preserve the output of `pip freeze` in a versioned lock file such as `requirements-lock.txt`.

## Installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running the pipeline

### 1. Configure the raw-data path

Create a symbolic link to the local data-collection directory:

```bash
ln -s /path/to/data_collection data/daily_raw
```

Raw image data are not required when using the occupancy labels already included in the processed datasets.

### 2. Process individual camera datasets

```bash
python3 scripts/process_all_camera.py
```

### 3. Fuse Alpha and Beta camera labels

```bash
python3 baseline/fuse_alpha_beta.py
```

### 4. Train and evaluate Random Forest

```bash
python3 baseline/train_fused_random_forest.py
```

### 5. Train and evaluate SGDClassifier

```bash
python3 baseline/train_fused_sgd.py
```

### 6. Run temporally ordered baselines

```bash
python3 baseline/train_rf_sliding_window.py
python3 baseline/train_sgd_sliding_window.py
```

## Generated outputs

Each fused training script creates a timestamped directory containing:

```text
analysis/<model>/<execution_timestamp>/
â”œâ”€â”€ figures/    # Confusion matrices, metric plots, and statistical analyses
â”œâ”€â”€ metrics/    # CSV files containing sample- and episode-level results
â””â”€â”€ model/      # Serialized trained model
```

The main metrics files are:

- `fused_random_forest_metrics.csv`
- `fused_sgd_metrics.csv`
- `episode_analysis.csv`
- `false_negative_analysis.csv`
- `permutation_feature_importance.csv` for Random Forest

## Privacy considerations

Cameras are retained for ground-truth generation, so the framework does not eliminate camera-related privacy risks. Privacy exposure is reduced because operational RF and SGD inference relies on environmental measurements rather than continuous image-based classification. Raw classroom images are excluded from this public repository; only the derived labels and environmental features needed to reproduce the machine-learning experiments are provided.

## Limitations

- The dataset was collected in a single classroom and may not represent other buildings or occupancy patterns.
- The classes are strongly imbalanced.
- YOLO-generated labels may contain detection errors and should be validated against a manually annotated subset.
- The 10-minute interval favors sustained occupancy and may miss short visits.
- Random train-test splitting may overestimate generalization because of temporal autocorrelation.
- Runtime measurements are hardware- and workload-dependent.

## Academic context

This repository supports a master's research project on multimodal sensing, edge computing, machine learning, and temporal evaluation for indoor occupancy detection.

## License

This project is distributed under the terms of the repository's `LICENSE` file.
