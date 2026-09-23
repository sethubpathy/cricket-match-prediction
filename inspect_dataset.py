import os
import pandas as pd

# Check relative and absolute paths
candidate_paths = [
    os.path.abspath("data/raw/matches.csv"),
    os.path.abspath("cricket-match-prediction/data/raw/matches.csv"),
    r"C:\Users\RC\Desktop\CRICKET RESULT PREDICTION PROJECT\data\raw\matches.csv",
    r"C:\Users\RC\Desktop\CRICKET RESULT PREDICTION PROJECT\cricket-match-prediction\data\raw\matches.csv",
]

file_path = None
for p in candidate_paths:
    if os.path.exists(p):
        file_path = p
        break

if not file_path:
    raise FileNotFoundError("Could not find matches.csv in candidate locations.")

df = pd.read_csv(file_path)

print("=" * 60)
print("CRICKET DATASET INSPECTION REPORT")
print("=" * 60)
print(f"Actual dataset file path: {file_path}")
print(f"Number of matches: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

teams = sorted(list(set(df["team1"].dropna()).union(set(df["team2"].dropna()))))
print(f"Number of teams: {len(teams)}")

venues = df["venue"].nunique()
print(f"Number of venues: {venues}")

print("\nMissing values:")
null_counts = df.isnull().sum()
for col, cnt in null_counts.items():
    pct = (cnt / len(df)) * 100
    if cnt > 0:
        print(f"  - {col}: {cnt} missing ({pct:.2f}%)")
    else:
        print(f"  - {col}: 0 missing")

print("\nAvailable columns:")
for i, col in enumerate(df.columns, 1):
    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "None"
    dtype = df[col].dtype
    print(f"  {i:2d}. {col:<18} (dtype: {str(dtype):<7}, sample: {sample})")
print("=" * 60)
