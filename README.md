# AI / Machine Learning Internship Projects

This repository contains my work for the 4-week AI/ML internship program. Each week has its own folder with code, documentation and results.

| Week | Task | Folder | Status |
|---|---|---|---|
| 1 | [Week 1 | [`week1/`](./week1) | Completed |
| 2 | [Week 2  | [`week2/`](./week2) | Completed |
| 3 | [Week 3 ] | [`week3/`](./week3) | Completed |
| 4 | AI Project Deployment & Capstone | [`week4/`](./week4) | Completed |

## Repository structure
```
.
├── README.md
├── week1/        
├── week2/        
├── week3/        
└── week4/        # Breast Cancer Diagnosis Predictor (Flask deployment)
    ├── app.py
    ├── train.py
    ├── models/
    ├── templates/
    ├── tests/
    ├── reports/
    ├── requirements.txt
    └── Dockerfile
```

---

## Week 1: [Task title]
**Objective:** [Copy the objective from the task page.]

**What I did**
- [Step / technique 1]
- [Step / technique 2]
- [Step / technique 3]

**Tools & libraries:** [e.g. Python, Pandas, NumPy, Matplotlib]

**Key results:** [Main findings, metrics or outputs]

**How to run**
```bash
cd week1
pip install -r requirements.txt
python [main_file].py
```

---

## Week 2: [Task title]
**Objective:** [Copy the objective from the task page.]

**What I did**
- [Step / technique 1]
- [Step / technique 2]
- [Step / technique 3]

**Tools & libraries:** [list]

**Key results:** [Main findings, metrics or outputs]

**How to run**
```bash
cd week2
pip install -r requirements.txt
python [main_file].py
```

---

## Week 3: [Task title]
**Objective:** [Copy the objective from the task page.]

**What I did**
- [Step / technique 1]
- [Step / technique 2]
- [Step / technique 3]

**Tools & libraries:** [list]

**Key results:** [Main findings, metrics or outputs]

**How to run**
```bash
cd week3
pip install -r requirements.txt
python [main_file].py
```

---

## Week 4: AI Project Deployment & Capstone
### Breast Cancer Diagnosis Predictor

**Objective:** Build an end-to-end machine learning project (preprocessing, training, testing, deployment) served through Flask, with model serialization using Pickle/Joblib, a prediction API, documentation and a presentation.

> Educational project only. Not a medical device.

**Dataset:** Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features, no missing values), loaded from scikit-learn.

**What I did**
- Stratified 80/20 train-test split (455 / 114 samples).
- Built a scikit-learn `Pipeline` (StandardScaler + classifier) to prevent data leakage.
- Compared Logistic Regression, SVM (RBF) and Random Forest with 5-fold cross-validation.
- Tuned the best model (Logistic Regression, C = 0.1) with GridSearchCV.
- Serialized the pipeline with **Joblib** and **Pickle**.
- Deployed it as a **Flask** REST API with a web interface and input validation.
- Wrote 7 automated tests and added a Dockerfile (gunicorn) for production.

**Results (held-out test set)**

| Metric | Score |
|---|---|
| Accuracy | 97.4% |
| Precision | 97.3% |
| Recall | 98.6% |
| F1-score | 97.9% |
| ROC-AUC | 0.996 |

**API endpoints**

| Method | Route | Description |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| GET | `/features` | Feature names and ranges |
| POST | `/predict` | Send 30 features as JSON, get prediction and confidence |

Example response:
```json
{"prediction": "malignant", "class_index": 0, "confidence": 1.0,
 "probabilities": {"benign": 0.0, "malignant": 1.0}}
```
Invalid input returns HTTP 400 with an error message.

**How to run**
```bash
cd week4
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python train.py                  # trains and saves the model
python -m unittest discover -s tests -v
python app.py                    # open http://localhost:5000
```

**Docker**
```bash
docker build -t cancer-predictor .
docker run -p 5000:5000 cancer-predictor
```

**Deliverables:** project report (Word), presentation (PowerPoint), source code and documentation.

**Future work:** threshold tuning for higher malignant recall, SHAP explainability, CI/CD with automated tests, FastAPI migration.

---

## Setup (all weeks)
- Python 3.10+
- Install each week's dependencies with `pip install -r requirements.txt` inside its folder.

## Author
**[Your Name]**
[LinkedIn / Email]
