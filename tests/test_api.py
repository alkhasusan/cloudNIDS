
import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "data" / "MachineLearningCSV" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(csv_path, low_memory=False)
df.columns = df.columns.str.strip()

row = df.iloc[0]
features = row.drop(labels=["Label"], errors="ignore")
payload = {"features": features.to_dict()}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print("Dataset label:", row.get("Label", "Unknown"))
print("API status:", response.status_code)
print("API response:", response.json())
