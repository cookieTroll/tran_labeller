# Transaction Analysis System
## Overview
An ETL pipeline for personal finance transaction labeling and analysis, demonstrating good data engineering practices and principles.

## Key Features
- Automated transaction data processing and labeling
- Data validation and quality checks
- Configurable labeling rules engine
- Jupyter notebooks for exploratory data analysis

## Technical Highlights
- Clean architecture with separation of concerns
- Data validation and schema enforcement
- Modular and extensible design
- Comprehensive testing suite
- Configurable data processing pipelines
- Documentation following industry standards

## Project Structure
```
project_root/ 
├── data/ 
│ ├── raw/ # Raw transaction data 
│ └── processed/ # Cleaned and labeled data 
├── src/ # Source code 
├── notebooks/ # Jupyter notebooks for analysis 
├── tests/ # Test suite 
├── config/ # Configuration files 
└── docs/ # Documentation
``` 

## Installation
```
bash
# Clone the repository
git clone [your-repo-url]
# Create and activate virtual environment
python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
``` 

## Usage
1. Place raw (as exported from bank account) transaction data in csv format in `data/raw/`
2. Configure input in 'config/generic' (e.g. `input_config_airbank.yaml`)
3. Configure output in 'config/generic' (e.g. `output_config.yaml`)
4. Set up labeling rules in `config/generic`
5. Create a high level config in `config` (e.g. `file_config_air.yaml`), where you basically specify directiories for required configs
6. Run the pipeline - open transaction_analysis.ipynb, specify path to the config and run.
7. The workflow consists from 2 steps - after keyword matching a file with categorization conflicts is generated in location specified in point 5. Open it and for each line select 1 of the offered options (or write your own entirely). After you save the file, run second part.

## Development
```bash
# Run tests
pytest

# Generate documentation
cd docs
make html
```

