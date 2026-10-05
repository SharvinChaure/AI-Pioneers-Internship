# Breast Cancer Diagnosis Predictor — ML Deployment Capstone

End-to-end machine learning project: data preprocessing → model training → evaluation → serialization (Joblib/Pickle) → deployment as a **Flask** prediction API with a web UI.

> Educational project only — not a medical device.

## Results (held-out test set, 114 samples)
| Metric | Score |
|---|---|
| Accuracy | 97.4% |
| Precision | 97.3% |
| Recall | 98.6% |
| F1 | 97.9% |
| ROC-AUC | 0.996 |

Model: StandardScaler + Logistic Regression (C=0.1), chosen via 5-fold CV over Logistic Regression, Random Forest and SVM, then tuned with GridSearchCV.

## Project structure
```
app.py              Flask API + web UI
train.py            Training / evaluation / serialization pipeline
models/             model.joblib, model.pkl, metadata.json
templates/          index.html (web form)
tests/test_api.py   API unit tests
reports/            Evaluation figures
Dockerfile, requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python train.py          # trains and saves models/
python -m unittest discover -s tests -v
python app.py            # http://localhost:5000
```

## API
| Method | Route | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| GET | `/features` | Feature names and ranges |
| POST | `/predict` | JSON with the 30 features → prediction |

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample.json
# {"prediction":"malignant","class_index":0,"confidence":1.0,"probabilities":{"benign":0.0,"malignant":1.0}}
```
Invalid input returns HTTP 400 with an error message.

## Docker / production
```bash
docker build -t cancer-predictor . && docker run -p 5000:5000 cancer-predictor
```
Uses gunicorn; deployable to Render, Railway, Heroku, or any container host.
