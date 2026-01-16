# Temporal Robustness of Vision-Based Occupancy Models

Este repositório contém a pipeline de processamento, treinamento e avaliação temporal de modelos de Machine Learning para predição de ocupação, utilizando dados provenientes de sensores ambientais multimodais e câmeras processadas com YOLO.

O foco principal do projeto é avaliar a robustez temporal dos modelos, analisando como o desempenho evolui à medida que novos dados semanais são incorporados ao treinamento.

# 📌 Objetivos do Projeto

Integrar dados multimodais de sensores ambientais e câmeras.
Relacionar dados sensoriais com labels de ocupação extraídos das câmeras.
Treinar modelos supervisionados de forma cumulativa ao longo do tempo.
Avaliar a estabilidade, saturação e generalização temporal dos modelos.
Comparar desempenho semana a semana (accuracy e F1-score).

# 🗂 Estrutura do Projeto

temporal-robustness-vision-models/
│
├── config.py
│
├── data/
│ ├── daily_raw/ # Symlink para o repositório de coleta
│ │ └── YYYY-MM-DD/
│ │ ├── camera_alpha/
│ │ ├── camera_beta/
│ │ ├── webcam_usb/
│ │ └── \*.csv
│ │
│ ├── processed/
│ │ └── <camera>/
│ │ ├── week_39.csv
│ │ ├── week_40.csv
│ │ └── ...
│ │
│ └── results/
│ └── <camera>\_weekly_cumulative.csv
│
├── models/
│ └── <camera>/
│ ├── week_39_cumulative.joblib
│ ├── week_40_cumulative.joblib
│ └── ...
│
├── scripts/
│ ├── load_camera.py
│ ├── load_sensors.py
│ ├── build_dataset.py
│ ├── process_all_camera.py
│ └── train_all_cameras.py

# 📥 Dados de Entrada

**Sensores Ambientais**

Cada sensor exporta um CSV diário contendo:

timestamp (epoch, segundos)
average
std_dev
conf_interval_lower
conf_interval_upper
No treinamento, apenas a coluna average é utilizada.

# Câmeras

Cada câmera possui sua própria pasta diária, contendo:
Imagens capturadas
CSV com:
timestamp (epoch)
pessoas
ocupada (label binária: 1 ocupado, 0 não ocupado)
As câmeras são tratadas de forma isolada, gerando modelos independentes por câmera.

# 🔄 Processamento dos Dados

1. Normalização Temporal
   Todos os timestamps são convertidos de epoch para datetime
   Os dados são agregados em janelas de 10 minutos

2. Integração Multimodal

Para cada dia:
O CSV da câmera é carregado (fonte do label ocupada)
Os CSVs dos sensores são mesclados pelo timestamp
Dados ausentes são preenchidos com zero

3. Agregação Semanal

Os dias são agrupados em semanas consecutivas

Cada semana gera um arquivo:
data/processed/<camera>/week_XX.csv

# 🤖 Treinamento dos Modelos

Algoritmo: Random Forest Classifier

**Treinamento cumulativo:**

Semana 39 → treino apenas com semana 39
Semana 40 → treino com semanas 39 + 40
Semana 41 → treino com semanas 39 + 40 + 41
etc.

**Avaliação**

O modelo treinado até a semana n é avaliado apenas nos dados da semana n

**Métricas:**

Accuracy
F1-score (com zero_division=0)

# 📊 Resultados

Os resultados são salvos em:
data/results/<camera>\_weekly_cumulative.csv

**Esses resultados permitem analisar:**

Saturação de desempenho
Impacto da distribuição das classes
Robustez temporal do modelo

# 🔧 Instalação das Dependências

Requisitos do Sistema

Python 3.10 ou superior

**1. Criar ambiente virtual (recomendado)**
python3 -m venv venv
source venv/bin/activate

**2. Atualizar o pip**
pip install --upgrade pip

**3. Instalar dependências do projeto**
pip install -r requirements.txt

# ▶️ Como Executar

**1. Criar o symlink para os dados de coleta**
ln -s /path/para/data_collection data/daily_raw

**2. Processar os dados**
python scripts/process_all_camera.py

**3. Treinar e avaliar os modelos**
python scripts/train_all_cameras.py

# 🎓 Contexto Acadêmico

Este projeto faz parte de um trabalho de mestrado, com foco em:

Sensoriamento multimodal.
Visão computacional aplicada.
Robustez temporal de modelos de Machine Learning.
Predição de ocupação em ambientes inteligentes.
