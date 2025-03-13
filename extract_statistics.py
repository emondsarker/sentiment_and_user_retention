import pandas as pd
import os
import glob
import warnings

# Define dataset directory
dataset_dir = "datasets"

# Function to load CSV files
def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    if os.path.exists(file_path):
        print(f"Loading {file_path} ...")
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None

# Function to extract specific content-related statistics
def extract_content_statistics(df, name):
    """Extracts post, answer, comment counts, and unique post counts from the dataset."""
    if df is None or df.empty:
        return {name: "File not found or empty"}

    stats = {}

    # Validate required columns exist before processing
    if 'content_type' in df.columns:
        try:
            stats['content_type_counts'] = df['content_type'].value_counts().to_dict()
        except Exception as e:
            print(f"Error calculating content_type counts for {name}: {e}")
            stats['content_type_counts'] = {}
    
    if 'post_id' in df.columns and 'content_type' in df.columns:
        try:
            stats['total_posts'] = df[df['content_type'] == 'Post']['post_id'].nunique()
        except Exception as e:
            print(f"Error calculating total posts for {name}: {e}")
            stats['total_posts'] = 0
    
    if 'parent_post_id' in df.columns and 'content_type' in df.columns:
        try:
            stats['total_answers'] = df[df['content_type'] == 'Answer']['post_id'].nunique()
        except Exception as e:
            print(f"Error calculating total answers for {name}: {e}")
            stats['total_answers'] = 0
    
    if 'content_type' in df.columns:
        try:
            stats['total_comments'] = df[df['content_type'] == 'Comment']['post_id'].nunique()
        except Exception as e:
            print(f"Error calculating total comments for {name}: {e}")
            stats['total_comments'] = 0
    
    return {name: stats}

# Function to extract full summary statistics
def extract_full_statistics(df, name):
    """Extracts full summary statistics for all columns in the dataset."""
    if df is None or df.empty:
        return {name: "File not found or empty"}

    try:
        summary = df.describe(include="all")  # Compute full summary statistics
        output_file = os.path.join(dataset_dir, f"{name}_summary.csv")
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Warning: Overwriting existing summary file {output_file}")
        
        summary.to_csv(output_file)  # Save the summary to a CSV file
        print(f"Summary statistics for {name} saved to {output_file}")
        return {name: summary}
    except Exception as e:
        print(f"Error creating summary statistics for {name}: {e}")
        return {name: "Error creating summary statistics"}

# Process all CSV files in the dataset directory
all_stats = {}
all_summary_stats = {}

# Make sure the dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory {dataset_dir} does not exist")
    exit(1)

# Get all CSV files in the dataset directory
csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))

if not csv_files:
    print(f"No CSV files found in {dataset_dir} directory")
else:
    for file_path in csv_files:
        # Skip any summary files that might already be in the directory
        if '_summary.csv' in file_path:
            print(f"Skipping existing summary file: {file_path}")
            continue
            
        # Extract filename without extension to use as the dataset name
        file_name = os.path.basename(file_path)
        name = os.path.splitext(file_name)[0]
        
        # Clean up filename if necessary
        name = name.replace('_posts_with_comments_answers', '')
        
        # Load and process the dataset
        df = load_csv(file_path)
        if df is not None:
            content_stats = extract_content_statistics(df, name)
            full_stats = extract_full_statistics(df, name)
            
            all_stats.update(content_stats)
            all_summary_stats.update(full_stats)
        else:
            print(f"Skipping {file_path} due to loading error")

    # Save extracted content-related statistics to a CSV file
    try:
        content_stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
        content_stats_output_file = os.path.join(dataset_dir, "content_statistics.csv")
        
        # Check if file already exists
        if os.path.exists(content_stats_output_file):
            print(f"Warning: Overwriting existing content statistics file {content_stats_output_file}")
            
        content_stats_df.to_csv(content_stats_output_file)
        print(f"\nExtracted Content Statistics for {len(csv_files)} datasets saved to {content_stats_output_file}")
    except Exception as e:
        print(f"Error saving content statistics: {e}")
