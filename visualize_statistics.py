import pandas as pd
import matplotlib.pyplot as plt

# Extracted statistics from dataset
# Update this with actual extracted stats from extract_statistics.py
dataset_stats = {
    "macos": {"Post": 15923, "Answer": 10959, "Comment": 5899},
    "mongodb": {"Post": 64100, "Answer": 45087, "Comment": 17656},
    "linux": {"Post": 245901, "Answer": 215062, "Comment": 103836},
}

# Convert dictionary to DataFrame
df_stats = pd.DataFrame(dataset_stats).T

# Print dataset statistics as a table
print("\nDataset Statistics Table:")
print(df_stats.to_string())

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
df_stats.plot(kind="bar", ax=ax)

# Formatting the plot
ax.set_title("Content Type Distribution Across Datasets", fontsize=14)
ax.set_ylabel("Count", fontsize=12)
ax.set_xlabel("Dataset", fontsize=12)
ax.legend(title="Content Type")
plt.xticks(rotation=0)

# Show the plot
plt.show()
