import os
import pandas as pd

# Your specific directory paths
input_dir = r"C:\Users\thene\github-classroom\sentiment_and_user_retention\datasets"
output_dir = os.path.join(input_dir, "processed_for_senti4sd")
os.makedirs(output_dir, exist_ok=True)

# List of your specific files
target_files = [
    "updated_chrome_os_with_authors.csv",
    "updated_linux_posts_with_authors.csv",
    "updated_matlab_posts_with_comments_answers.csv",
    "updated_unity_posts_with_comments_answers.csv",
    "updated_xcode_posts_with_comments_answers.csv"
]

for fname in target_files:
    fpath = os.path.join(input_dir, fname)
    print(f"🔄 Processing: {fname}")

    if os.path.exists(fpath):
        # Read original file
        df = pd.read_csv(fpath, dtype={"post_title": str}, low_memory=False)

        # Create the dataframe for Senti4SD with both IDs
        senti4sd_df = pd.DataFrame()
        senti4sd_df['ID'] = range(1, len(df) + 1)  # Sequential IDs for Senti4SD
        senti4sd_df['post_id'] = df['post_id']  # Keep original post_id
        
        # Combine text fields
        senti4sd_df['Text'] = df.apply(
            lambda row: f"{row['content_type']}: {row['post_title']} {row['content']}".strip(),
            axis=1
        )

        # Save ID, post_id, and Text columns
        outpath = os.path.join(output_dir, fname)
        senti4sd_df.to_csv(outpath, index=False)
        print(f"✅ Processed: {fname}")
        print(f"   Created file with {len(senti4sd_df)} rows")
    else:
        print(f"⚠️ File not found: {fname}")

print("\n💾 All files processed. Check output in:", output_dir)
