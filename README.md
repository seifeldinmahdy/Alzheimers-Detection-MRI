# Alzheimer's Detection from MRI using Deep Learning & Metaheuristic Optimization

A comprehensive deep learning pipeline for classifying Alzheimer's disease stages from MRI brain scans using the OASIS dataset. This project features **metaheuristic optimization algorithms** for hyperparameter tuning and **Explainable AI (XAI)** for model interpretability.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements an end-to-end machine learning pipeline for detecting and classifying Alzheimer's disease into four stages:

| Class | Description |
|-------|-------------|
| **Non-Demented** | No signs of dementia |
| **Very Mild Dementia** | Early stage symptoms |
| **Mild Dementia** | Moderate cognitive decline |
| **Moderate Dementia** | Significant impairment |

## Key Features

- **EfficientNet-B0 Backbone** - Transfer learning with pretrained weights
- **ACO Feature Selection** - Ant Colony Optimization for optimal feature subset
- **Metaheuristic Hyperparameter Optimization** - DE, GWO, PSO, BAT, WOA algorithms
- **Simulated Annealing Fine-tuning** - SA-tuned optimization for best performers
- **Explainable AI** - GradCAM and SHAP visualizations for model interpretability
- **Mixed Precision Training** - AMP for faster GPU training

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 86.57% |
| **Balanced Accuracy** | 90.20% |
| **Macro F1-Score** | 89.33% |
| **Best Validation F1** | 88.01% |

### Per-Class F1 Scores

| Class | F1-Score |
|-------|----------|
| Mild Dementia | 91.72% |
| Moderate Dementia | **100.00%** |
| Non-Demented | 85.09% |
| Very Mild Dementia | 80.52% |

### Metaheuristic Algorithm Ranking

| Rank | Algorithm | Score | Time (s) |
|------|-----------|-------|----------|
| 1 | **GWO** | 0.7949 | 315.84 |
| 2 | PSO | 0.7849 | 284.01 |
| 3 | BAT | 0.7731 | 359.03 |
| 4 | DE | 0.7629 | 380.66 |
| 5 | GWO (SA-tuned) | 0.6798 | 76.22 |
| 6 | WOA (SA-tuned) | 0.6162 | 67.43 |

## Project Structure

```
Alzheimers-Detection-MRI/
├── nic-project.ipynb              # Main notebook with complete pipeline
├── LICENSE                        # MIT License
├── README.md
└── results/
    ├── aco_results.json           # ACO feature selection results
    ├── config.json                # Configuration parameters
    ├── final_metrics.json         # Final model performance metrics
    ├── metaheuristic_ranking.csv  # Algorithm comparison rankings
    ├── phase1_results.csv         # Phase 1 optimization results
    ├── phase2_algo_param_tuning.csv # Phase 2 SA tuning results
    ├── xai_params_best.json       # Optimized XAI parameters
    ├── logs/
    │   └── training_history.csv   # Training metrics over epochs
    ├── plots/                     # Generated visualizations
    └── xai_outputs/               # Explainability outputs
        ├── summary.html           # XAI summary report
        ├── Mild_Dementia/
        ├── Moderate_Dementia/
        ├── Non_Demented/
        └── Very_mild_Dementia/
```

## Pipeline Stages

### 1. Data Preparation
- Load OASIS MRI dataset
- Apply balanced subset sampling
- Data augmentation and preprocessing

### 2. ACO Feature Selection
- Ant Colony Optimization for feature subset selection
- Optimized pheromone-based search

### 3. Phase 1: Metaheuristic Optimization
Compare multiple optimization algorithms:
- **DE** - Differential Evolution
- **GWO** - Grey Wolf Optimizer
- **PSO** - Particle Swarm Optimization
- **BAT** - Bat Algorithm

### 4. Phase 2: SA-Tuned Optimization
- Simulated Annealing fine-tuning of top performers
- WOA (Whale Optimization Algorithm) and Grey Wolf Optimization (GWO) integration

### 5. XAI Analysis
- GradCAM heatmaps for visual explanations
- SHAP values for feature importance
- Per-class explanation outputs

### 6. Final Training
- Train with optimized hyperparameters
- Early stopping with patience
- Comprehensive metric logging

## Configuration

Key hyperparameters (from `config.json`):

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Backbone | EfficientNet-B0 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Dropout | 0.3 |
| Mixed Precision | Enabled |

## Usage

### Running on Kaggle (Recommended)

1. **Create a new Kaggle notebook**
2. **Add the dataset:**
   - Search for `ninadaithal/imagesoasis`
   - Add to your notebook
3. **Enable GPU:**
   - Settings → Accelerator → GPU T4 ×2
4. **Upload and run `nic-project.ipynb`**

### Local Setup

```bash
# Clone the repository
git clone https://github.com/seifeldinmahdy1/Alzheimers-Detection-MRI.git
cd Alzheimers-Detection-MRI

# Install dependencies
pip install torch torchvision timm shap numpy pandas matplotlib scikit-learn

# Run the notebook
jupyter notebook nic-project.ipynb
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- timm
- SHAP
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- CUDA-capable GPU (recommended)

## Dataset

This project uses the [OASIS Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) containing brain MRI scans labeled with dementia severity levels.

**Dataset Distribution (Balanced Subset):**
- Mild Dementia: 1,800 images
- Moderate Dementia: 488 images
- Non-Demented: 2,912 images
- Very Mild Dementia: 2,000 images

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OASIS](https://www.oasis-brains.org/) for the brain imaging dataset
- Kaggle for GPU compute resources
- The open-source community for optimization algorithm implementations

---
