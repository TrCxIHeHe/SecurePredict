# SecurePredict — Windows setup

## 1. Extract the starter project

Extract `SecurePredict_v0_1.zip` somewhere convenient, e.g.:

```text
Documents\SecurePredict
```

Open that folder in VS Code.

## 2. Create a virtual environment

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Download both datasets from UCI

```powershell
python src/data/download_datasets.py
```

Expected folders:

```text
data\raw\ai4i_2020\
    features.csv
    targets.csv

data\raw\rt_iot2022\
    features.csv
    targets.csv
```

## 4. Run the first audit

```powershell
python src/data/inspect_datasets.py
```

Save the terminal output or copy it into the project report later. We want to inspect:

- shapes
- feature names
- target names
- missing values
- duplicate rows
- class distributions

## 5. Launch the placeholder dashboard

```powershell
streamlit run app/app.py
```

This is only the V0.1 shell. The actual models will be wired in after the audit.

## Next milestone

After the dataset audit, we will build leakage-safe preprocessing and baseline models before any tuning. Do not manually rebalance the entire dataset before the train/test split; any oversampling must happen inside the training pipeline only.
