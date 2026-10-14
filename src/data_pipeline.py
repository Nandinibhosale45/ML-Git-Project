from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "student_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed_student_data.csv"
REPORT_FILE = ROOT / "reports" / "pipeline_report.txt"

print("=== END-TO-END DATA PIPELINE ===")

# 1. Data collection
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

data = pd.read_csv(INPUT_FILE)
original_rows = len(data)
print(f"1. Data collection: {original_rows} rows loaded")

# 2. Data preprocessing
data.columns = data.columns.str.strip().str.lower()
required = [
    "student_id", "study_hours", "attendance",
    "assignments", "previous_score", "marks"
]

missing_columns = [c for c in required if c not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

data = data[required].copy()

for column in required:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.drop_duplicates()
data = data.dropna(subset=["student_id"])

for column in required:
    if column != "student_id":
        data[column] = data[column].fillna(data[column].median())

print(f"2. Preprocessing: {len(data)} rows after cleaning")

# 3. Feature engineering
data["average_score"] = (
    data["previous_score"] + data["marks"]
) / 2

data["performance_band"] = data["average_score"].apply(
    lambda score: "Good" if score >= 60 else "Needs Improvement"
)

print("3. Feature engineering: average_score and performance_band created")

# 4. Data validation
valid = (
    data["study_hours"].ge(0)
    & data["attendance"].between(0, 100)
    & data["assignments"].ge(0)
    & data["previous_score"].between(0, 100)
    & data["marks"].between(0, 100)
)

invalid_rows = int((~valid).sum())
data = data.loc[valid].copy()

if data.empty:
    raise ValueError("No valid records remain after validation.")

print(f"4. Data validation: {len(data)} valid rows, {invalid_rows} invalid rows removed")

# 5. Save processed data
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_FILE, index=False)

print(f"5. Processed dataset saved: {OUTPUT_FILE.name}")

# 6. Documentation and report
report = f"""
END-TO-END DATA PIPELINE REPORT
================================
Input file: data/student_data.csv
Original rows: {original_rows}
Rows after duplicate removal and missing-ID handling: {len(data) + invalid_rows}
Invalid rows removed during validation: {invalid_rows}
Final valid rows: {len(data)}
Output file: data/processed_student_data.csv
Features created: average_score, performance_band
Status: SUCCESS
"""

REPORT_FILE.write_text(report.strip() + "\n", encoding="utf-8")
print(f"6. Report saved: {REPORT_FILE.relative_to(ROOT)}")
print("PIPELINE EXECUTED SUCCESSFULLY")
