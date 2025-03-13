import pandas as pd
import os
import glob
import warnings
import re
import argparse
import json

# Define dataset directory
dataset_dir = "datasets"
summary_dir = os.path.join(dataset_dir, "summaries")
os.makedirs(summary_dir, exist_ok=True)  # Create summaries directory if it doesn't exist

# Function to load CSV files with better date handling
def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame with improved date parsing."""
    if os.path.exists(file_path):
        print(f"Loading {file_path} ...")
        try:
            # Load without date parsing first
            df = pd.read_csv(file_path, low_memory=False)
            
            # If post_creation_date exists, try to convert it
            if 'post_creation_date' in df.columns:
                # Try explicit date formats
                date_formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%m/%d/%Y %H:%M:%S',
                    '%m/%d/%Y',
                ]
                
                # Print a sample of dates to help debugging
                sample = df['post_creation_date'].dropna().head(3).tolist()
                if sample:
                    print(f"Sample dates: {sample}")
                    
                # Try each format
                for date_format in date_formats:
                    try:
                        print(f"Trying date format: {date_format}")
                        df['post_creation_date'] = pd.to_datetime(df['post_creation_date'], format=date_format, errors='coerce')
                        valid_count = df['post_creation_date'].notna().sum()
                        if valid_count > 0:
                            print(f"Successfully parsed {valid_count} dates with format {date_format}")
                            return df
                    except Exception as e:
                        print(f"Format {date_format} failed: {e}")
                
                # If all formats fail, use the generic parser
                print("Using generic date parser as fallback")
                df['post_creation_date'] = pd.to_datetime(df['post_creation_date'], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None

# Function to extract specific content-related statistics
def extract_content_statistics(df, name):
    """Extracts post, answer, comment counts, and additional text-based statistics from the dataset."""
    if df is None or df.empty:
        return {"status": "File not found or empty"}

    stats = {}
    
    # Basic dataset information
    stats['total_records'] = len(df)
    stats['columns'] = list(df.columns)

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
    
    # Extract text-based statistics from the content column
    if 'content' in df.columns:
        try:
            stats['average_word_count'] = round(df['content'].apply(lambda x: len(str(x).split())).mean(), 2)
            stats['average_char_count'] = round(df['content'].apply(lambda x: len(str(x))).mean(), 2)
            stats['max_post_length'] = df['content'].apply(lambda x: len(str(x))).max()
            stats['min_post_length'] = df['content'].apply(lambda x: len(str(x))).min()
            stats['num_questions'] = df['content'].apply(lambda x: str(x).count('?')).sum()
            stats['num_code_snippets'] = df['content'].apply(lambda x: str(x).count('<code>')).sum()
            stats['num_links'] = df['content'].apply(lambda x: str(x).count('<a>')).sum()
        except Exception as e:
            print(f"Error processing content statistics for {name}: {e}")
    
    # Extract post frequency statistics from post_creation_date
    if 'post_creation_date' in df.columns:
        try:
            # Count valid dates
            valid_dates_count = df['post_creation_date'].notna().sum()
            stats['valid_dates_count'] = valid_dates_count
            stats['invalid_dates_count'] = len(df) - valid_dates_count
            
            if valid_dates_count > 0:
                valid_dates_df = df[df['post_creation_date'].notna()]
                stats['earliest_post_date'] = valid_dates_df['post_creation_date'].min()
                stats['latest_post_date'] = valid_dates_df['post_creation_date'].max()
                
                if pd.api.types.is_datetime64_dtype(valid_dates_df['post_creation_date']):
                    year_counts = valid_dates_df.groupby(valid_dates_df['post_creation_date'].dt.year)['post_id'].count()
                    stats['posts_per_year'] = year_counts.to_dict()
            else:
                print(f"No valid dates found in {name}")
        except Exception as e:
            print(f"Error processing post frequency statistics for {name}: {e}")
    
    return stats

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Extract statistics from dataset files.')
    parser.add_argument('--files', nargs='+', help='Specific files to process (e.g., "linux.csv mongodb.csv")')
    args = parser.parse_args()
    
    # Process specific dataset files or all CSV files
    all_stats = {}
    
    # Make sure the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        exit(1)
    
    # Define default files to process if no files are specified
    default_files = [
        "linux.csv", 
        "oracle_database_posts_with_comments_answers.csv", 
        "macos.csv", 
        "mongodb.csv", 
        "sqlite_posts_with_comments_answers.csv"
    ]
    
    # Get list of files to process
    if args.files:
        files_to_process = args.files
    else:
        files_to_process = default_files
        print(f"No files specified, using default list: {', '.join(default_files)}")
    
    # Process each specified file
    processed_files = []
    for file_name in files_to_process:
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            # Extract filename without extension to use as the dataset name
            name = os.path.splitext(file_name)[0]
            
            # Clean up filename if necessary
            name = name.replace('_posts_with_comments_answers', '')
            
            # Load and process the dataset
            df = load_csv(file_path)
            if df is not None:
                # Process the statistics
                stats = extract_content_statistics(df, name)
                
                # Save individual summary file (both CSV and JSON for flexibility)
                individual_stats_df = pd.DataFrame([stats])
                individual_stats_df.index = [name]
                
                csv_summary_file = os.path.join(summary_dir, f"{name}_summary.csv")
                individual_stats_df.to_csv(csv_summary_file)
                
                # Save as JSON for better readability of nested structures
                json_summary_file = os.path.join(summary_dir, f"{name}_summary.json")
                with open(json_summary_file, 'w') as f:
                    json.dump(stats, f, indent=4, default=str)
                
                print(f"Individual summaries for {name} saved to {csv_summary_file} and {json_summary_file}")
                
                # Add to combined stats
                all_stats[name] = stats
                processed_files.append(file_path)
            else:
                print(f"Skipping {file_path} due to loading error")
        else:
            print(f"File not found: {file_path}")
    
    # Save extracted content-related statistics to a CSV file
    if all_stats:
        try:
            content_stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
            content_stats_output_file = os.path.join(dataset_dir, "selected_content_statistics.csv")
            
            # Also save as JSON for better readability of nested structures
            json_output_file = os.path.join(dataset_dir, "selected_content_statistics.json")
            with open(json_output_file, 'w') as f:
                json.dump(all_stats, f, indent=4, default=str)
            
            # Check if file already exists
            if os.path.exists(content_stats_output_file):
                print(f"Warning: Overwriting existing statistics file {content_stats_output_file}")
                
            content_stats_df.to_csv(content_stats_output_file)
            print(f"\nExtracted Content Statistics for {len(processed_files)} datasets saved to:")
            print(f"- CSV: {content_stats_output_file}")
            print(f"- JSON: {json_output_file}")
        except Exception as e:
            print(f"Error saving content statistics: {e}")
    else:
        print("No statistics were collected, nothing to save.")

if __name__ == "__main__":
    main()
