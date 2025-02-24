# House Price Prediction AI

This project predicts house prices based on neighborhood sale prices, bank valuations, and property features.

## 🚀 How to Run

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Start the FastAPI Server
```sh
python -m uvicorn api.main:app --reload
```

This will start the API at: **http://127.0.0.1:8000**

### 3️⃣ Test the API

- Open your browser and go to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to access the Swagger UI.
- You can test different API endpoints from there.

### 4️⃣ Manually Call the API

You can use `curl` or Postman to test the API. Example request:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"feature1": value1, "feature2": value2, ...}'
```

### 5️⃣ Folder Structure
```
📂 house-price-prediction
│── 📂 api            # FastAPI application
│    ├── main.py      # API routes
│── 📂 src            # Core ML logic
│    ├── train.py     # Model training
│    ├── preprocess.py # Data preprocessing
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
```

### 6️⃣ Deployment
To deploy on a cloud service like **Render, AWS, or Heroku**, follow their Python deployment guides.

---

Now you're all set to run and improve your House Price Prediction AI! 🚀

