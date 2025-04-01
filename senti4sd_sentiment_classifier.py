import pandas as pd
import subprocess
import os
from tqdm import tqdm
import time
from IPython.display import clear_output

def print_status(message):
    """Print status message with timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

# Define paths and parameters
print_status("Initializing Senti4SD classifier...")

# Define directories using Windows path format
base_dir = os.getcwd()
datasets_dir = os.path.join(base_dir, 'datasets', 'preprocessed_for_senti4sd')
output_dir = os.path.join(base_dir, 'datasets', 'predictions')
jar_path = os.path.join(base_dir, 'external', 'pySenti4SD', 'java', 'Senti4SD.jar')
dsm_path = os.path.join(base_dir, 'external', 'pySenti4SD', 'java', 'dsm.bin')

# Convert to Windows path format
datasets_dir = datasets_dir.replace('/', '\\')
output_dir = output_dir.replace('/', '\\')
jar_path = jar_path.replace('/', '\\')
dsm_path = dsm_path.replace('/', '\\')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of datasets to process
datasets = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
# Get list of already processed files
processed_files = [f.replace('senti4sd_', '') for f in os.listdir(output_dir) if f.startswith('senti4sd_')]

print_status(f"Found {len(datasets)} datasets to process")
print_status(f"Already processed: {len(processed_files)} datasets")

# Process each dataset
for idx, dataset in enumerate(datasets, 1):
    # Skip if already processed
    if dataset in processed_files:
        print_status(f"⏭️ Skipping already processed: {dataset}")
        continue
        
    print_status(f"\n🔄 [{idx}/{len(datasets)}] Running Senti4SD on: {dataset}")
    
    input_file = os.path.join(datasets_dir, dataset)
    output_file = os.path.join(output_dir, f"senti4sd_{dataset}")
    
    # Modify the command line to include additional Java memory settings
    cmd = f'java -Xmx24g -XX:+UseG1GC -XX:+UseStringDeduplication -jar "{jar_path}" -i "{input_file}" -F A -W "{dsm_path}" -oc "{output_file}" -vd 600'
    
    print_status(f"📤 Command: {cmd}")
    
    try:
        # Run the command
        process = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Check if output file was created
        if os.path.exists(output_file):
            print_status(f"✅ Successfully processed {dataset}")
        else:
            print_status(f"⚠️ Warning: Output file not found: {output_file}")
            
    except subprocess.CalledProcessError as e:
        print_status(f"❌ Error with {dataset}:")
        print_status(e.stderr)
        continue
    except Exception as e:
        print_status(f"❌ Unexpected error with {dataset}:")
        print_status(str(e))
        continue
    
    # Optional: Clear output between datasets
    time.sleep(2)  # Pause to let user see completion message
    clear_output(wait=True)

# Final summary
print_status("\n📋 Processing Summary:")
for dataset in datasets:
    output_file = os.path.join(output_dir, f"senti4sd_{dataset}")
    status = "✅ Completed" if os.path.exists(output_file) else "❌ Failed"
    print(f"{dataset}: {status}")

print_status(f"\n💾 Results saved in: {output_dir}")