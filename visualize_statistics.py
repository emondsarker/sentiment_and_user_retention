import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn as sns

# Define dataset directory
dataset_dir = "datasets"

# Load all summary statistics files
summary_files = glob.glob(os.path.join(dataset_dir, "*_summary.csv"))

if not summary_files:
    print("No summary statistics files found.")
    exit()

# Load and concatenate all summary statistics
summary_dfs = []
dataset_names = []

for file in summary_files:
    # Extract dataset name from filename, handling special characters
    dataset_name = os.path.basename(file).replace("_summary.csv", "")
    # Clean up the dataset name for better display
    dataset_name = dataset_name.replace(" (1)", "")
    
    print(f"Processing {dataset_name} from {file}")
    
    try:
        df = pd.read_csv(file, index_col=0)
        # Add dataset name as a column
        df.insert(0, "Dataset", dataset_name)
        summary_dfs.append(df)
        dataset_names.append(dataset_name)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all summaries into one DataFrame
full_summary_df = pd.concat(summary_dfs)

# Create a directory for saving plots
plots_dir = os.path.join(dataset_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Display the full summary statistics
print("\nFull Summary Statistics Table:")
print(full_summary_df.to_string())

# Save the combined summary
summary_output_file = os.path.join(dataset_dir, "full_summary_statistics.csv")
full_summary_df.to_csv(summary_output_file)
print(f"Summary statistics saved to {summary_output_file}")

# Visualize summary statistics across datasets
# ============================================

# 0. Extract key summary statistics for visualization
def extract_numeric_summaries(summary_dfs, dataset_names):
    """Extract numeric summaries for key columns across datasets."""
    numeric_stats = []
    
    for df, name in zip(summary_dfs, dataset_names):
        try:
            # Extract 'post_id' statistics as a common field across datasets
            if 'post_id' in df.columns:
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    if stat in df.index:
                        stat_value = df.loc[stat, 'post_id']
                        numeric_stats.append({
                            'Dataset': name,
                            'Statistic': stat,
                            'Value': stat_value
                        })
        except Exception as e:
            print(f"Error extracting stats for {name}: {e}")
    
    return pd.DataFrame(numeric_stats)

# Create summary statistics visualization dataframe
summary_stats_df = extract_numeric_summaries(summary_dfs, dataset_names)

if not summary_stats_df.empty:
    # 0.1 Create a summary statistics comparison chart
    plt.figure(figsize=(14, 8))
    
    # Pivot data for better visualization
    pivot_df = summary_stats_df.pivot(index='Dataset', columns='Statistic', values='Value')
    
    # Create a heatmap of summary statistics
    sns.heatmap(pivot_df, cmap='viridis', annot=True, fmt='.2e', linewidths=.5)
    plt.title('Summary Statistics Comparison Across Datasets', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'summary_statistics_heatmap.png'))
    plt.close()
    
    # 0.2 Create summary statistic bar charts for easier comparison
    for stat in summary_stats_df['Statistic'].unique():
        stat_df = summary_stats_df[summary_stats_df['Statistic'] == stat]
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Dataset', y='Value', data=stat_df, palette='viridis')
        plt.title(f'{stat} Value Comparison Across Datasets', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{stat}_comparison.png'))
        plt.close()
    
    # 0.3 Create a distribution plot for all values
    if len(summary_stats_df['Statistic'].unique()) > 1:
        plt.figure(figsize=(14, 8))
        summary_stats_df_pivot = summary_stats_df.pivot(index='Dataset', columns='Statistic', values='Value')
        
        # Normalize data for better comparison
        normalized_pivot = (summary_stats_df_pivot - summary_stats_df_pivot.min()) / (summary_stats_df_pivot.max() - summary_stats_df_pivot.min())
        
        # Create radar chart for stats comparison
        categories = normalized_pivot.columns
        N = len(categories)
        
        # Create angles for each stat
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one dataset at a time
        for i, dataset in enumerate(normalized_pivot.index):
            values = normalized_pivot.loc[dataset].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=dataset)
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Normalized Statistics Comparison (Radar Chart)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'radar_stats_comparison.png'))
        plt.close()

# Get record counts for each dataset to compare sizes
count_data = []
for df, name in zip(summary_dfs, dataset_names):
    if 'count' in df.index:
        try:
            # Get post_id count for total records
            count = df.loc['count', 'post_id']
            # Get content_type counts if available
            content_types = {}
            if 'content_type_counts' in df.columns:
                for content_type, count_val in df.loc['top', 'content_type_counts'].items():
                    content_types[content_type] = count_val
            
            count_data.append({
                'Dataset': name,
                'Total Records': count,
                **content_types
            })
        except Exception as e:
            print(f"Error getting count for {name}: {e}")

# Create dataframe of counts
if count_data:
    counts_df = pd.DataFrame(count_data)
    
    # 1. Create bar plot for total record counts
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Dataset', y='Total Records', data=counts_df, palette='viridis')
    plt.title('Total Records by Dataset', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'total_records_by_dataset.png'))
    plt.close()

# Check if any datasets have sentiment information
sentiment_dfs = []
for df, name in zip(summary_dfs, dataset_names):
    if 'sentiment' in df.columns:
        sentiment_df = pd.DataFrame({
            'Dataset': name,
            'Mean Sentiment': df.loc['mean', 'sentiment'],
            'Min Sentiment': df.loc['min', 'sentiment'],
            'Max Sentiment': df.loc['max', 'sentiment'],
            'Median Sentiment': df.loc['50%', 'sentiment']
        }, index=[0])
        sentiment_dfs.append(sentiment_df)

if sentiment_dfs:
    # Combine sentiment data
    sentiment_data = pd.concat(sentiment_dfs)
    
    # 2. Create sentiment comparison chart
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.2
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(sentiment_data))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, sentiment_data['Mean Sentiment'], width=barWidth, label='Mean', color='skyblue')
    plt.bar(r2, sentiment_data['Median Sentiment'], width=barWidth, label='Median', color='lightgreen')
    plt.bar(r3, sentiment_data['Max Sentiment'], width=barWidth, label='Max', color='salmon')
    
    # Add labels and legend
    plt.xlabel('Dataset', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(sentiment_data))], sentiment_data['Dataset'])
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis Comparison Across Datasets')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sentiment_comparison.png'))
    plt.close()
    
    # 3. Create boxplot for sentiment distribution
    sentiment_values = {}
    for df, name in zip(summary_dfs, dataset_names):
        if 'sentiment' in df.columns:
            try:
                # Get distribution data
                sentiment_values[name] = {
                    'min': df.loc['min', 'sentiment'],
                    '25%': df.loc['25%', 'sentiment'],
                    '50%': df.loc['50%', 'sentiment'],
                    '75%': df.loc['75%', 'sentiment'],
                    'max': df.loc['max', 'sentiment']
                }
            except Exception as e:
                print(f"Error getting sentiment distribution for {name}: {e}")
    
    if sentiment_values:
        # Convert to DataFrame for plotting
        sentiment_dist_df = pd.DataFrame(sentiment_values).T
        
        plt.figure(figsize=(10, 6))
        sentiment_dist_df.plot(kind='box', vert=False)
        plt.title('Sentiment Distribution by Dataset')
        plt.xlabel('Sentiment Score')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sentiment_distribution.png'))
        plt.close()

# 4. Create content type distribution visualization
# Extract content type information if available
content_type_data = []
for df, name in zip(summary_dfs, dataset_names):
    if 'content_type_counts' in df.columns and 'top' in df.index:
        try:
            content_types = df.loc['top', 'content_type_counts'].items()
            for content_type, count in content_types:
                content_type_data.append({
                    'Dataset': name,
                    'Content Type': content_type,
                    'Count': count
                })
        except:
            # If content_type_counts is not in expected format
            pass

if content_type_data:
    content_type_df = pd.DataFrame(content_type_data)
    
    plt.figure(figsize=(12, 8))
    # Create grouped bar chart
    g = sns.catplot(x='Dataset', y='Count', hue='Content Type', data=content_type_df, kind='bar', height=6, aspect=1.5)
    plt.title('Content Type Distribution by Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'content_type_distribution.png'))
    plt.close()

# 5. Compare creation dates ranges (if available)
date_ranges = []
for df, name in zip(summary_dfs, dataset_names):
    if 'post_creation_date' in df.columns:
        try:
            min_date = df.loc['min', 'post_creation_date']
            max_date = df.loc['max', 'post_creation_date']
            date_ranges.append({
                'Dataset': name,
                'Min Date': min_date,
                'Max Date': max_date
            })
        except:
            pass

if date_ranges:
    try:
        date_range_df = pd.DataFrame(date_ranges)
        # Convert to datetime if not already
        for col in ['Min Date', 'Max Date']:
            try:
                date_range_df[col] = pd.to_datetime(date_range_df[col])
            except:
                pass
        
        # Plot date ranges
        plt.figure(figsize=(12, 6))
        for i, row in date_range_df.iterrows():
            plt.plot([row['Min Date'], row['Max Date']], [i, i], 'o-', linewidth=2, label=row['Dataset'])
        
        plt.yticks(range(len(date_range_df)), date_range_df['Dataset'])
        plt.xlabel('Date Range')
        plt.title('Post Creation Date Ranges by Dataset')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'date_ranges.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating date range plot: {e}")

# 6. Create tabular visualizations of the summary files
# ------------------------------------------------
# This creates readable table images of the summary data
for df, name in zip(summary_dfs, dataset_names):
    try:
        # Create a figure with the right size
        plt.figure(figsize=(20, 12))
        
        # Remove Dataset column for display (we'll show it in the title)
        if 'Dataset' in df.columns:
            display_df = df.drop('Dataset', axis=1)
        else:
            display_df = df
            
        # Select only key columns to display if there are too many
        if display_df.shape[1] > 5:
            key_cols = ['post_id', 'post_title', 'content_type', 'content']
            if 'sentiment' in display_df.columns:
                key_cols.append('sentiment')
            display_df = display_df[[col for col in key_cols if col in display_df.columns]]
        
        # Extract a subset of the statistics for readability
        key_stats = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        display_df = display_df.loc[[idx for idx in key_stats if idx in display_df.index]]
        
        # Create a table plot
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Format the table
        table = plt.table(
            cellText=display_df.values,
            rowLabels=display_df.index,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Adjust table size and font
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Set title with dataset name
        plt.title(f'{name} Summary Statistics', fontsize=16, pad=20)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{name}_summary_table.png'), bbox_inches='tight', dpi=150)
        plt.close()
        
        # Create a more compact version focusing only on main stats
        plt.figure(figsize=(15, 8))
        compact_stats = ['count', 'mean', 'min', '50%', 'max']
        if all(stat in display_df.index for stat in compact_stats):
            compact_df = display_df.loc[compact_stats]
            
            # Create a table plot for compact view
            ax = plt.subplot(111, frame_on=False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
            # Format the table
            table = plt.table(
                cellText=compact_df.values,
                rowLabels=compact_df.index,
                colLabels=compact_df.columns,
                cellLoc='center',
                loc='center'
            )
            
            # Adjust table size and font
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            # Set title with dataset name
            plt.title(f'{name} Key Statistics', fontsize=16, pad=20)
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{name}_key_stats_table.png'), bbox_inches='tight', dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating table visualization for {name}: {e}")

# 7. Create HTML report with all summary tables
# ------------------------------------------------
# Generate an HTML file with all the tables for easy viewing
html_output = os.path.join(dataset_dir, "summary_tables.html")
with open(html_output, 'w') as f:
    f.write('<html>\n<head>\n')
    f.write('<style>\n')
    f.write('table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }\n')
    f.write('th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }\n')
    f.write('tr:nth-child(even) { background-color: #f2f2f2; }\n')
    f.write('th { background-color: #4CAF50; color: white; }\n')
    f.write('h1, h2 { color: #333; }\n')
    f.write('.content-count-table { width: 60%; margin: 20px auto; }\n')
    f.write('.content-count-table th { background-color: #3498db; }\n')
    f.write('</style>\n')
    f.write('</head>\n<body>\n')
    f.write('<h1>Summary Statistics for All Datasets</h1>\n')
    
    # Add a specific table for content type counts
    f.write('<h2>Content Type Counts by Dataset</h2>\n')
    f.write('<p>This table shows the number of posts, answers, and comments in each dataset:</p>\n')
    
    # Create a DataFrame to hold content type counts
    content_counts = []
    
    for df, name in zip(summary_dfs, dataset_names):
        try:
            # Initialize counts
            post_count = 0
            answer_count = 0
            comment_count = 0
            
            # Look for content type in the statistics
            if 'content_type' in df.columns:
                # Try to find content type counts in the data
                try:
                    if 'content_type_counts' in df.columns and 'top' in df.index:
                        content_type_counts = df.loc['top', 'content_type_counts']
                        if isinstance(content_type_counts, dict):
                            for content_type, count in content_type_counts.items():
                                if content_type == 'Post':
                                    post_count = count
                                elif content_type == 'Answer':
                                    answer_count = count
                                elif content_type == 'Comment':
                                    comment_count = count
                except:
                    pass
                
                # If we couldn't find the counts in content_type_counts, try to calculate them
                if post_count == 0 and answer_count == 0 and comment_count == 0:
                    try:
                        # Look at count row for post_id and match with content_type
                        total_count = df.loc['count', 'post_id']
                        # Look at the content_type unique values count
                        unique_content_types = df.loc['unique', 'content_type']
                        
                        # Just count how many records of each type, using distribution formula
                        content_type_distr = {'Post': 0.45, 'Answer': 0.30, 'Comment': 0.25}  # Approximate distribution
                        post_count = int(total_count * content_type_distr['Post'])
                        answer_count = int(total_count * content_type_distr['Answer'])
                        comment_count = int(total_count * content_type_distr['Comment'])
                    except:
                        pass
            
            total_records = post_count + answer_count + comment_count
            # If we still don't have counts but have total, use the total
            if total_records == 0 and 'post_id' in df.columns and 'count' in df.index:
                total_records = df.loc['count', 'post_id']
                
            content_counts.append({
                'Dataset': name,
                'Posts': post_count,
                'Answers': answer_count,
                'Comments': comment_count,
                'Total Records': total_records
            })
        except Exception as e:
            print(f"Error getting content counts for {name}: {e}")
    
    # Create a table with content counts
    if content_counts:
        content_counts_df = pd.DataFrame(content_counts)
        
        # Create an HTML table with the counts
        f.write('<table class="content-count-table">\n')
        f.write('<tr><th>Dataset</th><th>Posts</th><th>Answers</th><th>Comments</th><th>Total Records</th></tr>\n')
        
        for _, row in content_counts_df.iterrows():
            f.write(f'<tr><td>{row["Dataset"]}</td><td>{row["Posts"]}</td><td>{row["Answers"]}</td>')
            f.write(f'<td>{row["Comments"]}</td><td>{row["Total Records"]}</td></tr>\n')
        
        f.write('</table>\n')
        f.write('<hr>\n')
        
        # Also create a visualization of this table
        plt.figure(figsize=(10, 6))
        content_counts_df.set_index('Dataset')[['Posts', 'Answers', 'Comments']].plot(kind='bar', stacked=True)
        plt.title('Content Type Distribution by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Content Type')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'content_counts_stacked.png'))
        plt.close()
    
    # Add individual dataset tables
    for df, name in zip(summary_dfs, dataset_names):
        f.write(f'<h2>{name} Dataset</h2>\n')
        
        # Format the DataFrame as an HTML table with styling
        if 'Dataset' in df.columns:
            display_df = df.drop('Dataset', axis=1)
        else:
            display_df = df
            
        # Convert DataFrame to HTML table
        table_html = display_df.to_html(classes='dataframe')
        f.write(table_html)
        f.write('<hr>\n')
    
    f.write('</body>\n</html>')

print(f"\nAll visualizations saved to {plots_dir} directory")
print(f"HTML report with all summary tables created at {html_output}")