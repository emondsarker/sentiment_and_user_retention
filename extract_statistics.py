import pandas as pd
import os

# Define dataset directory
dataset_dir = "dataset"  

# File paths
macos_path = os.path.join(dataset_dir, "macos.csv")
mongodb_path = os.path.join(dataset_dir, "mongodb.csv")
linux_path = os.path.join(dataset_dir, "linux.csv")

# Function to load CSV files
def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    if os.path.exists(file_path):
        print(f"Loading {file_path} ...")
        return pd.read_csv(file_path, low_memory=False)
    else:
        print(f"File not found: {file_path}")
        return None

# Load datasets
macos_df = load_csv(macos_path)
mongodb_df = load_csv(mongodb_path)
linux_df = load_csv(linux_path)

# Function to extract statistics
def extract_statistics(df, name):
    """Extracts post, answer, comment counts, and unique post counts from the dataset."""
    if df is None:
        return {name: "File not found or empty"}

    stats = {}

    # Count occurrences of different content types
    if 'content_type' in df.columns:
        stats['content_type_counts'] = df['content_type'].value_counts().to_dict()
    
    # Count unique posts (excluding answers/comments)
    if 'post_id' in df.columns:
        stats['total_posts'] = df[df['content_type'] == 'Post']['post_id'].nunique()
    
    # Count unique answers
    if 'parent_post_id' in df.columns:
        stats['total_answers'] = df[df['content_type'] == 'Answer']['post_id'].nunique()
    
    # Count unique comments (if applicable)
    if 'content_type' in df.columns and 'Comment' in df['content_type'].unique():
        stats['total_comments'] = df[df['content_type'] == 'Comment']['post_id'].nunique()
    
    return {name: stats}

# Extract statistics for each dataset
macos_stats = extract_statistics(macos_df, "macos")
mongodb_stats = extract_statistics(mongodb_df, "mongodb")
linux_stats = extract_statistics(linux_df, "linux")

# Print the results
print("Extracted Statistics:")
print(macos_stats)
print(mongodb_stats)
print(linux_stats)
