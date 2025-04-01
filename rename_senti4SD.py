import os
import pandas as pd

preprocessed_dir = r"C:\Users\thene\Documents\MiningData\preprocessed_for_senti4sd"

for fname in os.listdir(preprocessed_dir):
    if fname.endswith(".csv"):
        fpath = os.path.join(preprocessed_dir, fname)
        print(f"🔄 Updating column name in: {fname}")

        df = pd.read_csv(fpath)
        if 'combined_text' in df.columns:
            df.rename(columns={'combined_text': 'Text'}, inplace=True)
            df.to_csv(fpath, index=False)
            print(f"✅ Renamed column and saved: {fname}")
        else:
            print(f"⚠️ Skipped {fname} (no 'combined_text' column found)")
