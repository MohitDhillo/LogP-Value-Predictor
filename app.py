import streamlit as st
import requests
import json
from rdkit import Chem
import io
import os

# Try to import Draw module, but don't fail if it's not available
try:
    from rdkit.Chem import Draw
    DRAWING_AVAILABLE = True
except ImportError:
    DRAWING_AVAILABLE = False
    st.warning("Molecule visualization is not available in this environment. The app will still function for predictions.")

# Set page config
st.set_page_config(
    page_title="LogP Value Predictor",
    page_icon="🧪",
    layout="wide"
)

# API endpoint configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
st.sidebar.markdown(f"API URL: {API_URL}")

# Title and description
st.title("🧪 LogP Value Predictor")
st.markdown("""
This application predicts the LogP (octanol-water partition coefficient) value for molecules based on their SMILES representation.
""")

# Input section
st.header("Input Molecule")
smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g., CC(=O)O")

if smiles_input:
    try:
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES string. Please check your input.")
        else:
            # Display molecule structure if available
            if DRAWING_AVAILABLE:
                st.subheader("Molecule Structure")
                img = Draw.MolToImage(mol)
                # Create a container for the image with a specific width
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(img, caption="Molecule Structure", use_container_width=True)
            
            # Make prediction
            if st.button("Predict LogP"):
                with st.spinner("Making prediction..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"smiles": smiles_input},
                            timeout=10  # Add timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            st.header("Prediction Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Predicted LogP", f"{result['predicted_logp']:.2f}")
                            
                            with col2:
                                st.subheader("Molecular Descriptors")
                                for key, value in result['molecular_descriptors'].items():
                                    st.write(f"{key}: {value:.2f}")
                        else:
                            st.error(f"Error: {response.json()['detail']}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: Unable to connect to the prediction service. Please check if the API is running and accessible.")
                        st.error(f"Technical details: {str(e)}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    This model uses:
    - Morgan fingerprints for molecular representation
    - Molecular descriptors (molecular weight, TPSA, etc.)
    - Machine learning algorithms for prediction
    
    The model was trained on a dataset of molecules with experimentally determined LogP values.
    """)

# Add example SMILES
with st.expander("Example SMILES"):
    st.markdown("""
    Here are some example SMILES strings you can try:
    
    - Acetic acid: `CC(=O)O`
    - Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
    - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    """) 