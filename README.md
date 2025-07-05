# Credit Risk Audit Tool

Automated auditing toolkit for credit-risk prediction models, evaluating:

- **Explainability**: SHAP, LIME, feature importance  
- **Robustness**: adversarial attacks, noise injection, boundary testing  
- **Regulatory Compliance**: data-leakage checks, outlier detection  
- **ProsperLoan Dataset Modules**: end-to-end preprocessing, feature engineering, and modeling tailored to the ProsperLoan data to use as inputs


## Project Structure

- `notebooks/`: Main notebooks to run the tool, including dedicated notebooks for the ProsperLoan dataset (data exploration, preprocessing, and modeling).
- `audit_tool/`: Python modules for data processing, modeling, and auditing (explainability and robustness).
- `data/raw/`: Original, unprocessed datasets.
- `data/processed/`: Datasets ready for modeling.
- `models/`: Trained models and related artifacts.
- `outputs/`: Generated audit reports and result visualizations.

## Requirements

This project uses a virtual environment to manage dependencies.

### Data & Model Placement
- **Raw data**: upload your original CSV files to `data/raw/`  
- **Processed data**: upload any cleaned or feature-engineered CSVs to `data/processed/`  
- **Models**: place ready-to-audit model files (`.pkl` or `.joblib`) in `models/`  

### Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate   
pip install -r requirements.txt

```

## Running the Project

With the virtual environment activated:

- Open the notebook for generating reports located in `notebooks/05_reporting.ipynb`.
- The notebook will call functions from the `audit_tool/` modules to handle the auditing.

There are also other modules and notebooks for data preprocessing and modelling over the 
ProsperLoan dataset and notebooks for carrying out individual tests regarding explainability and robustness.

## Running the Project

With the virtual environment activated, follow these steps to run the full audit pipeline:

1. **Execute Notebooks in Order**  
   Each notebook builds on the previous one. Open and run cells top-to-bottom:
   - `notebooks/01_preprocessing.ipynb`  
     Load raw CSV(s) from `data/raw/`, clean, impute, engineer features, and save to `data/processed/`.  
   - `notebooks/02_modeling.ipynb`  
     Train and tune models on the processed ProsperLoan data; save best estimators to `models/`.  
   - `notebooks/03_explainability.ipynb`  
     Compute SHAP & LIME explanations and visualize feature importance.  
   - `notebooks/04_robustness.ipynb`  
     Run adversarial attacks, noise injections, boundary and label-flip tests.  
   - `notebooks/05_reporting.ipynb`  
     Generate final HTML reports by calling `audit_tool/reporting.py` and Jinja2 templates.

2. **Run Individual Components**  
   - If you only need explainability analyses, run **03_explainability.ipynb**.  
   - For robustness checks alone, open **04_robustness.ipynb**.  

3. **Verify Data & Models**  
   - Ensure your raw CSVs are in `data/raw/`.  
   - Confirm processed CSVs (if precomputed) are in `data/processed/`.  
   - Check that your pickle/joblib models to audit reside in `models/`.  

4. **View Outputs**  
   After running **05_reporting.ipynb**, the HTML files will appear in: `outputs/reports/explainability/` or `outputs/reports/robustness/`. Figures and confusion matrixes will be under `outputs/figures/`.  
