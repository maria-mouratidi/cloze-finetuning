# Cloze Fine-tuning Project

**Improving GPT-2 with Human Next-Word Prediction**

This project implements fine-tuning of GPT-2 models on human next-word prediction (Cloze) datasets to better approximate human language prediction patterns. Based on the research conducted at Tilburg University by Maria Mouratidi under the supervision of Bruno Nicenboim adn Giovanni Casanni.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This research investigates whether fine-tuning GPT-2 on Cloze datasets (human next-word predictions) can improve the model's ability to predict human reading patterns. The project includes:

- **Data Processing**: Loading and preprocessing Cloze datasets from multiple sources
- **Model Training**: Fine-tuning GPT-2 with special token handling
- **Evaluation**: Perplexity-based comparison between pre-trained and fine-tuned models
- **Visualization**: Comprehensive plots and statistical analysis

### Key Research Questions

1. Can we improve GPT-2 with human next-word prediction data?
2. Does the use of special tokens (EOS) improve model performance?
3. How does the fine-tuned model generalize to different datasets?

## Project Structure

```
cloze-finetuning/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py            # ClozeDataLoader class
│   │   └── utils.py             # Data utility functions
│   ├── models/                   # Model training
│   │   ├── __init__.py
│   │   ├── config.py            # Training configuration
│   │   └── trainer.py           # ClozeTrainer class
│   ├── evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   ├── perplexity.py        # Perplexity calculation
│   │   ├── comparison.py        # Model comparison
│   │   └── visualization.py     # Plotting utilities
│   ├── utils/                    # General utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Project configuration
│   │   ├── logging.py           # Logging setup
│   │   └── paths.py             # Path management
│   └── __init__.py
├── scripts/                      # Execution scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── run_pipeline.py          # Complete pipeline
├── config/                       # Configuration files
│   └── default.yaml             # Default configuration
├── datasets/                     # Original datasets
│   ├── cloze/
│   ├── peelle/
│   └── provo/
├── Code/                         # Original notebooks (for reference)
├── notebooks/                    # Clean analysis notebooks
├── outputs/                      # Generated outputs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

#### Training Parameters

```bash
python scripts/train.py \
    --datasets cloze peelle provo \
    --epochs 6 \
    --batch-size 1 \
    --max-samples 2000 \
    --special-tokens \
    --output-dir outputs/gpt2_cloze_model
```

#### Evaluation Parameters

```bash
python scripts/evaluate.py \
    --pretrained-model gpt2 \
    --finetuned-model outputs/gpt2_cloze_model \
    --datasets cloze gpt2 wikipedia \
    --max-samples 1000 \
    --create-plots \
    --output-dir outputs/evaluation_results
```

## Datasets

The project supports three main Cloze datasets:

### 1. Cloze Dataset
- **Source**: Primary Cloze dataset
- **Files**: `clozetrain_clean.csv`, `clozetest_clean.csv`
- **Description**: Human next-word predictions

### 2. Peelle Dataset
- **Source**: Peelle et al. (2020)
- **Files**: `peelle_clean.csv`
- **Description**: Completion norms for English sentence contexts

### 3. Provo Dataset
- **Source**: Provo Corpus
- **Files**: `provo_clean.csv`
- **Description**: Predictability norms from reading data

### External Datasets
- **GPT-2 Original Data**: For comparison with original training data
- **Wikipedia Data**: For generalization testing (loaded automatically)

## Results

### Key Findings

Based on the original research:

1. **Special Token Impact**: Using EOS tokens at the beginning of sentences improves model performance on sensible sentences
2. **Fine-tuning Effects**: The fine-tuned model shows mixed results, with potential overfitting to Cloze data
3. **Generalization**: Limited generalization to Wikipedia data suggests overfitting

### Evaluation Metrics

- **Perplexity**: Primary evaluation metric
- **Statistical Analysis**: Mean, median, standard deviation
- **Visualization**: Histogram comparisons and quantile tables

### Core Classes

#### ClozeDataLoader
```python
from src.data import ClozeDataLoader

loader = ClozeDataLoader(data_dir="datasets")
data = loader.load_cloze_data("cloze", clean=True)
combined = loader.combine_datasets(["cloze", "peelle"], sample_size=1000)
```

#### ClozeTrainer
```python
from src.models import ClozeTrainer, TrainingConfig

config = TrainingConfig(num_train_epochs=6, output_dir="./my_model")
trainer = ClozeTrainer(config)
trainer.train(train_texts, eval_texts)
```

#### PerplexityEvaluator
```python
from src.evaluation import PerplexityEvaluator

evaluator = PerplexityEvaluator()
perplexity = evaluator.calculate_perplexity("gpt2", encodings)
results = evaluator.evaluate_texts("./my_model", texts)
``