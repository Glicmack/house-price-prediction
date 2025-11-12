# House Price Prediction System

A machine learning project that predicts house prices in Indian real estate market using Random Forest regression. The system includes both a web interface and REST API for predictions.

## 🎯 Project Overview

This project demonstrates end-to-end ML workflow including:
- Data preprocessing and feature engineering
- Model training with scikit-learn pipelines
- Web application deployment with Flask
- REST API for integration with other services

## 📊 Dataset

The model is trained on house price data with the following features:

**Numeric Features:**
- Amount (in rupees)
- Carpet Area
- Floor
- Bathroom count
- Balcony count
- Car Parking spaces
- Super Area

**Categorical Features:**
- Location
- Status (Ready to move, Under construction, etc.)
- Transaction type (New/Resale)
- Furnishing (Furnished, Semi-furnished, Unfurnished)
- Facing direction
- Overlooking (Park, Main road, etc.)
- Ownership type

## 🚀 Features

- **Intelligent Price Parsing**: Handles Indian price formats (42 Lac, 1.2 Cr)
- **Robust Preprocessing**: Handles missing values and text-in-numeric fields
- **Log-transformed Target**: Better prediction for wide price ranges
- **Two Deployment Options**: 
  - Web UI for manual predictions
  - REST API for programmatic access

## 📁 Project Structure

```
.
├── train.py              # Training script with EDA and model evaluation
├── save_model.py       # Script to save trained model and metadata
├── app.py              # Flask web application
├── app_api.py          # Flask REST API
├── model.pkl           # Trained model (generated)
├── metadata.json       # Model metadata (generated)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

Run python save_model.py to generate the model locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python save_model.py
```

This will generate `model.pkl` and `metadata.json` files.

## 💻 Usage

### Web Application

Run the Flask web app:
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser and fill out the form.

### REST API

Run the API server:
```bash
python app_api.py
```

Make predictions via POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "Amount(in rupees)": "42 Lac",
    "Carpet Area": 630,
    "Floor": 3,
    "Bathroom": 2,
    "Balcony": 1,
    "Car Parking": 1,
    "Super Area": 950,
    "location": "Thane",
    "Status": "Ready to move",
    "Transaction": "Resale",
    "Furnishing": "Semi-Furnished",
    "facing": "North",
    "overlooking": "Park",
    "Ownership": "Freehold"
  }'
```

Response:
```json
{
  "pred_rupees": 4250000.0,
  "pred_formatted": "42.50 Lac (₹ 42,50,000)",
  "input_received": {...}
}
```

## 📈 Model Performance

- **Algorithm**: Random Forest Regressor (200 estimators)
- **Preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical
- **Target Transformation**: Log1p transformation for better prediction
- **Evaluation Metrics**: 
  - R² Score: ~0.85-0.90 (varies by dataset)
  - MAE: Typically within 5-10 Lac rupees
  - RMSE: Detailed in training output

Run `python train.py` to see full model evaluation including feature importances.

## 🔧 Configuration

Before deploying to production:

1. Change the Flask secret key in both apps:
```python
app.secret_key = "your-production-secret-key"
```

2. Disable debug mode:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

3. Use a production WSGI server (e.g., Gunicorn):
```bash
pip install gunicorn
gunicorn app:app
```

## 🧪 Model Training Details

The training pipeline (`train.py` and `save_model.py`) includes:

1. **Data Cleaning**:
   - Price format conversion (Lac/Cr → rupees)
   - Numeric extraction from text fields
   - Missing value imputation

2. **Feature Engineering**:
   - Median imputation for numeric features
   - Mode imputation for categorical features
   - StandardScaler normalization

3. **Model Selection**:
   - Random Forest with 200 trees
   - Optional XGBoost comparison
   - Cross-validation ready

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/Glicmack)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
