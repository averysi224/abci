# Archetype Based Conformal Intervals (ABCI)

This repository provides the official implementation of the methods presented in our paper, *Reliable and Interpretable Visual Field Progression Prediction with Diffusion Models and Conformal Prediction*, which has been accepted for publication at MICCAI 2025. 

<img width="717" alt="miccai_overview" src="https://github.com/user-attachments/assets/013815cb-c3bd-4131-87ec-f2800c53d176" />


## Installation

1. Install the required dependencies:

   ```bash
   cd puq
   pip install -e .
   ```
   
## 📥 Download Dataset
To run the experiments, download the required datasets from the links below:
- **UW Dataset**: [Download](https://drive.google.com/file/d/1r12w4RsmaJEC_9mr2J5cCX9ZIXyMgWaY/view?usp=drive_link)
- **GRN Dataset**: [Download](https://drive.google.com/file/d/1iJEmjEcdKetcpEbK0JM0E9q0n9M98tyU/view?usp=drive_link)

unzip them in the 'puq/data' folder.

## 💌 Basic Usage
To run the script with default settings:
```sh
python run.py --no-cache --archetypes --alpha 0.25 
```

## 🏥 UW Dataset Experiments

### 🟢 Mild Case (q = 0.95, β = 0.1)
```sh
python run.py --no-cache --archetypes --data data/UW_subgroups/mild --alpha 0.25 --q 0.95 --beta 0.1 
```

### 🟡 Moderate Case (q = 0.9, β = 0.14)
```sh
python run.py --no-cache --archetypes --data data/UW_subgroups/moderate --alpha 0.25 --q 0.9 --beta 0.14
```

### 🔴 Severe Case (q = 0.9, β = 0.155)
```sh
python run.py --no-cache --archetypes --dataset data/UW_subgroups/severe --alpha 0.25 --q 0.9 --beta 0.155
```

---

## 🧠 GRN Dataset Experiments

### 🟢 Mild Case (q = 0.95, β = 0.1)
```sh
python run.py --no-cache --archetypes --data data/Scheie_subgroups/mild --alpha 0.25 --q 0.95 --beta 0.1
```

### 🟡 Moderate Case (q = 0.9, β = 0.15)
```sh
python run.py --no-cache --archetypes --data data/Scheie_subgroups/moderate --alpha 0.25 --q 0.9 --beta 0.15
```

### 🔴 Severe Case (q = 0.9, β = 0.2)
```sh
python run.py --no-cache --archetypes --data data/Scheie_subgroups/severe --alpha 0.25 --q 0.9 --beta 0.2
```

## 🚀 Additional Notes
- `--no-cache`: Ensures the script runs fresh without using cached results.
- `--archetypes`: Enables the archetypal analysis module.
- `--dataset`: Specifies the dataset (`UW` or `GRN`).
- `--q` and `--beta`: Control the model's sensitivity and regularization.

