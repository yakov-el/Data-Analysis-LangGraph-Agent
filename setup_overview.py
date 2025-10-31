import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from dotenv import load_dotenv

# --- הגדרות סביבה (חיוני לגישה ל-BigQuery) ---

# טען משתני סביבה מהקובץ .env אם יש (למשל, GOOGLE_API_KEY, למרות שהוא לא נחוץ כאן)
load_dotenv() 


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "langgraph-ecommerce-test-bbbaad28a1af.json"
# שימוש ב-ADC (מומלץ): לוודא שהופעל gcloud auth application-default login בטרמינל.

# ---------- CONFIG ----------
output_file = "overview.txt"
table_names = [
    "bigquery-public-data.thelook_ecommerce.orders",
    "bigquery-public-data.thelook_ecommerce.order_items",
    "bigquery-public-data.thelook_ecommerce.products",
    "bigquery-public-data.thelook_ecommerce.users"
]

# ---------- CLEANING FUNCTION ----------
def clean_for_json(obj: Any) -> Any:
    """הופך כל ערך לא נתמך (כגון NaT/NaN) למחרוזת או None."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        # אם יש ערך NaT (Not a Time), נחזיר None
        return None if pd.isna(obj) else obj.isoformat()
    elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
        # טיפול ב-NaN (Not a Number)
        return None
    elif pd.isna(obj):
        return None
    else:
        return obj

# ---------- MAIN FUNCTION ----------
def export_table_description(table_name: str, client: bigquery.Client) -> str:
    """מביא את סכמת הטבלה ודוגמה של 5 שורות."""
    try:
        table_ref = client.get_table(table_name)
        # שאילתה עם גירשי backtick כפולים עבור שם הטבלה המלא
        query = f"SELECT * FROM `{table_name}` LIMIT 5" 
        df = client.query(query).result().to_dataframe()

        # ניקוי ערכי NaT ו-NaN
        df = df.replace({pd.NaT: None, np.nan: None})
        sample_rows = clean_for_json(df.to_dict(orient="records"))

        description = f"Table: {table_ref.table_id}\n"
        description += f"Dataset: {table_ref.dataset_id}, Project: {table_ref.project}\n"
        description += f"Number of rows: {table_ref.num_rows}\n"
        description += "Columns:\n"
        for field in table_ref.schema:
            description += f"  - {field.name} ({field.field_type}, {field.mode})\n"
        description += "Sample rows (up to 5):\n"
        description += json.dumps(sample_rows, ensure_ascii=False, indent=2)
        description += "\n" + "="*80 + "\n\n"

        return description

    except Exception as e:
        return f"Failed to export table {table_name}: {e}\n" + "="*80 + "\n\n"

# ---------- EXPORT EXECUTION ----------
# יצירת קליינט BigQuery (משתמש בהרשאות המוגדרות בסביבה)
client = bigquery.Client()

with open(output_file, "w", encoding="utf-8") as f:
    for t in table_names:
        print(f"Processing table: {t}")
        desc = export_table_description(t, client)
        f.write(desc)

print(f"✅ Overview generation complete. Saved to {output_file}")