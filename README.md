# Application de prédiction des exacerbations
Dans le cadre de FPI

```bash
python run.py
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

```bash
conda remove -n ENV_NAME --all
```