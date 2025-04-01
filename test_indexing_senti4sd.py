# Create a test notebook or script to verify Senti4SD indexing

import pandas as pd
import subprocess
import os

# Define paths - using correct paths from your folder structure
jar_path = os.path.join('external', 'pySenti4SD', 'java', 'Senti4SD.jar')
dsm_path = os.path.join('external', 'pySenti4SD', 'java', 'dsm.bin')
test_file = "test_senti4sd.csv"
output_file = "test_output.csv"

# Create a small test file with known column positions
test_df = pd.DataFrame({
    'id': [1, 2, 3],
    'text': ['This is good', 'This is bad', 'This is neutral'],
    'other': ['a', 'b', 'c']
})

# Save test file
test_df.to_csv(test_file, index=False)

# Print the test file contents for verification
print("Test file contents:")
print(test_df)
print("\nTesting column indexing...")

# Test with 0-based index (column 1)
cmd_0 = f'java -jar "{jar_path}" -i "{test_file}" -F A -W "{dsm_path}" -oc "{output_file}" -vd 600 -f 1'
print("\nTrying 0-based indexing command:")
print(cmd_0)
try:
    result_0 = subprocess.run(cmd_0, shell=True, capture_output=True, text=True)
    print("Output:", result_0.stdout)
    print("Error:", result_0.stderr)
except Exception as e:
    print(f"Error running command: {e}")

# Test with 1-based index (column 2)
cmd_1 = f'java -jar "{jar_path}" -i "{test_file}" -F A -W "{dsm_path}" -oc "{output_file}" -vd 600 -f 2'
print("\nTrying 1-based indexing command:")
print(cmd_1)
try:
    result_1 = subprocess.run(cmd_1, shell=True, capture_output=True, text=True)
    print("Output:", result_1.stdout)
    print("Error:", result_1.stderr)
except Exception as e:
    print(f"Error running command: {e}")
