""" Test api application """

import pandas as pd
import requests
import json
from src.config.paths import RAW_DATA_DIR

# Load CSV
df = pd.read_csv(RAW_DATA_DIR / "test.csv")
df = df.drop(columns=["id"]).head()

# Convert to list-of-dicts (row-oriented)
payload = {
    "columns": df.to_dict(orient="list")
}
#print(payload)

# Send request
resp = requests.post(
    "http://localhost:8000/predict",
    json=payload,
    timeout=300
)

print(resp.status_code)
print(resp.json())
