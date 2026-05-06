# Cognitive Load Estimation Using Keystroke Dynamics 🧠⌨️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)

A professional Machine Learning pipeline that predicts human cognitive load levels by analyzing biometric keystroke dynamics and typing behavioral patterns.

## 🌟 Key Features

- **Robust ML Architecture:** Built using an Object-Oriented Programming (OOP) approach for maximum modularity and reusability.
- **Random Forest Classifier:** Utilizes a highly accurate ensemble learning method to classify cognitive states (e.g., Low, Medium, High).
- **Data Engineering Pipeline:** Includes automated dataset preprocessing, `StandardScaler` feature normalization, and `LabelEncoder` target encoding.
- **Model Serialization:** Automatically saves the trained model, scaler, and encoder (`.pkl`) for immediate deployment and real-time inference.
- **Automated Visualizations:** Generates and exports high-quality Confusion Matrix and Feature Importance charts automatically.

## 🛠️ Technology Stack

- **Language:** Python
- **Machine Learning:** `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`
- **Model Serialization:** `joblib`

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ridhi001/coginitive-load-estimation.git
   cd coginitive-load-estimation
   ```

2. **Install the required libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib joblib
   ```

3. **Run the pipeline:**
   ```bash
   python cognitive_load_estimation.py
   ```

## 📊 Outputs generated automatically:
When you run the script, it will train the model, evaluate it, and generate the following artifacts in your directory:
- `cognitive_load_model.pkl`: Your deployment-ready serialized model.
- `confusion_matrix_enhanced.png`: A visual breakdown of prediction accuracy.
- `feature_importance_enhanced.png`: A bar chart showing which keystroke metrics are most important for predicting cognitive load.

## 🔮 Predicting New Data (Inference)
The class is designed to easily accept new data points (simulating a user typing right now) and return their predicted cognitive load instantly, making it highly suitable for backend integration.

```python
from cognitive_load_estimation import CognitiveLoadModel
import numpy as np

# Simulate reading 6 new typing features from a live user
sample_data = np.array([[40, 150, 90, 450, 0.08, 5]])

# The model instantly outputs their cognitive state
pipeline = CognitiveLoadModel()
# ... 
```
