# Credit Risk Audit Tool

Automated auditing tool for credit risk prediction models, evaluating explainability, robustness, and regulatory compliance.

## Project Structure

- `notebooks/`: Main notebook to run the full pipeline.
- `audit_tool/`: Python modules for data processing, modeling, and auditing.
- `data/raw/`: Original, unprocessed datasets.
- `data/processed/`: Datasets ready for modeling.
- `models/`: Trained models and related artifacts.
- `reports/`: Generated audit reports and result visualizations.

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

- Open the main notebook located in `notebooks/main.ipynb`.
- The notebook will call functions from the `audit_tool/` modules to handle data preprocessing, model training, and auditing.
