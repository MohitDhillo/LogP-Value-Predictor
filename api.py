from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import joblib
import os

app = FastAPI(title="Molecular Property Prediction API",
             description="API for predicting LogP values from SMILES strings",
             version="1.0.0")

class MoleculeInput(BaseModel):
    smiles: str

def validate_smiles(smiles):
    """Validate SMILES string and return RDKit molecule object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None

def get_morgan_fingerprint(mol, radius=2, nBits=1024):
    """Generate Morgan fingerprint for a molecule"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def get_molecular_descriptors(mol):
    """Calculate basic molecular descriptors"""
    descriptors = {
        'MolWt': Descriptors.ExactMolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
    }
    return descriptors

# Load the trained model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    print("Warning: Model files not found. Please train the model first.")

@app.post("/predict")
async def predict_logp(molecule: MoleculeInput):
    """
    Predict LogP value for a given SMILES string
    """
    # Validate SMILES
    mol = validate_smiles(molecule.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    # Generate features
    morgan_fp = get_morgan_fingerprint(mol)
    descriptors = get_molecular_descriptors(mol)
    features = np.hstack([morgan_fp, list(descriptors.values())])
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return {
        "smiles": molecule.smiles,
        "predicted_logp": float(prediction),
        "molecular_descriptors": descriptors
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Molecular Property Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 