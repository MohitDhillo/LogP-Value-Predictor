# Molecular Property Predictor

A machine learning application for predicting LogP (octanol-water partition coefficient) values of molecules based on their SMILES representations. This project includes a FastAPI backend for model inference and a Streamlit frontend for an interactive user interface.

## Features

- SMILES string validation and molecule structure visualization
- LogP value prediction using machine learning models
- Molecular descriptor calculation and display
- Interactive web interface
- RESTful API for programmatic access

## Prerequisites

- Python 3.7+
- RDKit
- FastAPI
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LogP-Value-Predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
LogP-Value-Predictor/
├── README.md
├── requirements.txt
├── molecular_property_prediction.py  # Model training script
├── api.py                           # FastAPI backend
├── app.py                           # Streamlit frontend
├── model.joblib                     # Trained model (generated after training)
├── scaler.joblib                    # Feature scaler (generated after training)
└── logP_dataset.csv                 # Training dataset
```

## Usage

### 1. Train the Model

First, train the model using the provided dataset:
```bash
python molecular_property_prediction.py
```

This will:
- Train the Random Forest model
- Save the model as 'model.joblib'
- Save the feature scaler as 'scaler.joblib'
- Generate visualization plots

### 2. Start the Backend Server

In a terminal, start the FastAPI backend:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Launch the Frontend

In another terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

## Using the Application

1. Open the Streamlit interface in your web browser
2. Enter a SMILES string in the input field (or use one of the example SMILES provided)
3. The molecule structure will be displayed automatically
4. Click "Predict LogP" to get the prediction
5. View the predicted LogP value and molecular descriptors

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /`: Welcome message
- `POST /predict`: Predict LogP value for a given SMILES string
  ```json
  {
    "smiles": "CC(=O)O"
  }
  ```

## Example SMILES Strings

Try these example molecules:
- Acetic acid: `CC(=O)O`
- Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
- Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

## Model Details

The model uses:
- Morgan fingerprints for molecular representation
- Molecular descriptors (molecular weight, TPSA, etc.)
- Random Forest algorithm for prediction
- Standardized features using StandardScaler

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
