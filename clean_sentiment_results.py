import os
import time
import pandas as pd

def print_status(message):
    """Print a timestamped status message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def validate_indices(predictions_df, processed_df, pred_file):
    """Validate index relationships between prediction and processed DataFrames."""
    print_status(f"\nValidating indices for {pred_file}:")
    
    # Check Senti4SD output format
    print_status(f"Senti4SD results: {len(predictions_df)} rows")
    print_status(f"First few Senti4SD IDs: {', '.join(predictions_df['id'].head().tolist())}")
    
    # Check processed file format
    print_status(f"Input file: {len(processed_df)} rows")
    print_status(f"ID range in input: {processed_df['ID'].min()} to {processed_df['ID'].max()}")
    
    # Check post_id presence
    if 'post_id' not in processed_df.columns:
        raise ValueError("Input file missing 'post_id' column")
    
    print_status(f"Unique post_ids in input: {processed_df['post_id'].nunique()}")

# Define directories
base_dir = os.getcwd()
predictions_dir = os.path.join(base_dir, 'datasets', 'predictions')
clean_output_dir = os.path.join(base_dir, 'datasets', 'clean_predictions')
processed_dir = os.path.join(base_dir, 'datasets', 'processed_for_senti4sd')

# Create clean output directory
os.makedirs(clean_output_dir, exist_ok=True)

# Get list of prediction files and already cleaned files
prediction_files = [f for f in os.listdir(predictions_dir) if f.startswith('senti4sd_')]
cleaned_files = [f.replace('clean_', '') for f in os.listdir(clean_output_dir) if f.startswith('clean_')]

print_status(f"Found {len(prediction_files)} files to process")
print_status(f"Already cleaned: {len(cleaned_files)} files")

for pred_file in prediction_files:
    # Skip if already cleaned
    if pred_file in cleaned_files:
        print_status(f"⏭️ Skipping already cleaned: {pred_file}")
        continue
        
    print_status(f"\n🔄 Processing {pred_file}")
    
    try:
        # Read files
        pred_path = os.path.join(predictions_dir, pred_file)
        predictions_df = pd.read_csv(pred_path)
        
        processed_file = pred_file.replace('senti4sd_', '')
        processed_path = os.path.join(processed_dir, processed_file)
        processed_df = pd.read_csv(processed_path)
        
        # Validate data
        validate_indices(predictions_df, processed_df, pred_file)
        
        # Keep original detailed scores
        detailed_df = predictions_df[['id', 'Sim_subj', 'Sim_obj', 'Sim_pos', 'Sim_neg', 'User_Mention']]
        
        # Extract numeric part from Senti4SD IDs and add 1 to match input file IDs
        detailed_df['ID'] = detailed_df['id'].str.extract('t(\d+)').astype(int) + 1
        
        # Merge with processed file to get post_ids
        detailed_df = detailed_df.merge(
            processed_df[['ID', 'post_id']], 
            on='ID',
            how='left'  # Keep all Senti4SD results
        )
        
        # Organize columns
        final_cols = ['id', 'ID', 'post_id', 'Sim_subj', 'Sim_obj', 'Sim_pos', 'Sim_neg', 'User_Mention']
        detailed_df = detailed_df[final_cols]
        
        # Save cleaned results
        output_path = os.path.join(clean_output_dir, f"clean_{pred_file}")
        detailed_df.to_csv(output_path, index=False)
        
        # Print summary
        print_status(f"Results summary for {pred_file}:")
        print_status(f"Total Senti4SD results: {len(detailed_df)}")
        print_status(f"Results with matching post_ids: {detailed_df['post_id'].notna().sum()}")
        print_status(f"✅ Saved cleaned results to: {output_path}")
        
    except Exception as e:
        print_status(f"❌ Error processing {pred_file}: {str(e)}")
        continue

print_status("\n📋 Processing Summary:")
for pred_file in prediction_files:
    output_path = os.path.join(clean_output_dir, f"clean_{pred_file}")
    status = "✅ Completed" if os.path.exists(output_path) else "❌ Failed"
    print(f"{pred_file}: {status}")

print_status(f"\n💾 All processing complete. Results saved in: {clean_output_dir}")
