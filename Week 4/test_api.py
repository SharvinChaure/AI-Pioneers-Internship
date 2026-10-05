import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.datasets import load_breast_cancer
from app import app

data = load_breast_cancer(as_frame=True)

def sample(i):
    return {k: float(v) for k, v in data.data.iloc[i].items()}

class APITests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["status"], "ok")

    def test_home_page(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_predict_malignant(self):
        i = int(data.target[data.target == 0].index[0])
        r = self.client.post("/predict", json=sample(i))
        self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["prediction"], "malignant")

    def test_predict_benign(self):
        i = int(data.target[data.target == 1].index[0])
        j = self.client.post("/predict", json=sample(i)).get_json()
        self.assertEqual(j["prediction"], "benign"); self.assertTrue(0.5 <= j["confidence"] <= 1)

    def test_missing_features(self):
        self.assertEqual(self.client.post("/predict", json={"mean radius": 10}).status_code, 400)

    def test_bad_values(self):
        s = sample(0); s["mean radius"] = "abc"
        self.assertEqual(self.client.post("/predict", json=s).status_code, 400)

    def test_no_json(self):
        self.assertEqual(self.client.post("/predict", data="x").status_code, 400)

if __name__ == "__main__":
    unittest.main()
