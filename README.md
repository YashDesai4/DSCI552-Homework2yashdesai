# DSCI 552 - Homework 2

This repository contains a complete, reproducible solution for Homework 2.

## Files

- `Homework2_Yash_Desai.ipynb` - main submission with code, figures, and written answers
- `analysis.py` - reusable analysis functions used by the notebook
- `download_data.py` - downloads the official UCI workbook
- `data/Folds5x2_pp.xlsx` - expected dataset location (first worksheet is used)
- `requirements.txt` - Python dependencies

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py            # only needed if data/ is absent
jupyter notebook Homework2_Yash_Desai.ipynb
```

In Jupyter, select **Kernel > Restart Kernel and Run All Cells**. All paths are
relative to the repository, and the split is reproducible (`random_state=42`).

## Data source

P. Tufekci and H. Kaya, *Combined Cycle Power Plant*, UCI Machine Learning
Repository, 2014. DOI: https://doi.org/10.24432/C5002N (CC BY 4.0).

