# Ensemble ML Model — Early Liver Disease Detection

## Project Structure
```
liver_project/
├── data/
│   └── Indian Liver Patient Dataset (ILPD).csv   ← place dataset here
├── outputs/
│   ├── models/      ← saved .pkl model files
│   └── figures/     ← generated charts (PNG)
├── main.py           ← run this to train everything
├── predict.py        ← run this to classify new patients
├── data_loader.py
├── preprocessing.py
├── balancing.py
├── model_training.py
├── evaluation.py
├── utils.py
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python main.py        # train all models
python predict.py     # classify example patients
```
