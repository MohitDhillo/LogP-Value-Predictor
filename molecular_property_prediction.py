import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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

def main():
    # Step 1: Data Collection and Initial Exploration
    print("Loading dataset...")
    df = pd.read_csv('logP_dataset.csv')
    
    print("\nDataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nLogP Statistics:")
    print(df['LogP'].describe())
    
    # Step 2: Data Preprocessing
    print("\nPreprocessing data...")
    df['molecule'] = df['SMILES'].apply(validate_smiles)
    df = df.dropna(subset=['molecule'])
    df = df.drop_duplicates(subset=['SMILES'])
    
    print(f"Number of valid molecules: {len(df)}")
    print(f"Number of unique molecules: {df['SMILES'].nunique()}")
    
    # Step 3: Feature Engineering
    print("\nGenerating features...")
    morgan_fps = np.array([get_morgan_fingerprint(mol) for mol in df['molecule']])
    descriptors = pd.DataFrame([get_molecular_descriptors(mol) for mol in df['molecule']])
    
    X = np.hstack([morgan_fps, descriptors])
    y = df['LogP'].values
    
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    
    # Step 4: Model Training and Evaluation
    print("\nTraining models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')
    
    # Train and save the best model (Random Forest)
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    joblib.dump(best_model, 'model.joblib')
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nBest Model (Random Forest) Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Step 5: Visualization and Analysis
    print("\nGenerating visualizations...")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual LogP')
    plt.ylabel('Predicted LogP')
    plt.title('Actual vs Predicted LogP Values')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'Morgan_{i}' for i in range(morgan_fps.shape[1])] + list(descriptors.columns),
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nModel and visualizations have been saved.")

if __name__ == "__main__":
    main() 