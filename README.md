# Application de prédiction des exacerbations
Dans le cadre de FPI

```bash
python -m streamlit run app.py
```


## Build

```bash
pyinstaller --onefile --additional-hooks-dir=./hooks run.py --clean
pyinstaller run.spec --clean
```

## Config

```bash
conda env create -f environment.yaml
```