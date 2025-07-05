# Credit Risk Audit Tool

Automated auditing tool for credit risk prediction models, evaluating explainability, robustness, and regulatory compliance.

## Project Structure

- `notebooks/`: Main notebooks to run the tool.
- `audit_tool/`: Python modules for data processing, modeling, and auditing (explainability and robustness).
- `data/raw/`: Original, unprocessed datasets.
- `data/processed/`: Datasets ready for modeling.
- `models/`: Trained models and related artifacts.
- `outputs/`: Generated audit reports and result visualizations.

## Requirements

This project uses a virtual environment to manage dependencies.  
To set up the environment:

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
ProsperLoan dataset and notebooks for seeing individual tests regarding explainability and robustness.
