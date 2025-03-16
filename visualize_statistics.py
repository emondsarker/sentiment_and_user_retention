import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn as sns
import warnings
import json
from matplotlib.gridspec import GridSpec

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define directories
dataset_dir = "datasets"
summary_dir = os.path.join(dataset_dir, "summaries")
plots_dir = os.path.join(dataset_dir, "new_rq_plots")
os.makedirs(plots_dir, exist_ok=True)

def load_summary_data():
    """Load the combined statistics file and return as DataFrame"""
    # Try loading from JSON first (better preserves data types)
    json_file = os.path.join(dataset_dir, "selected_content_statistics.json")
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Convert to DataFrame with datasets as rows
        df = pd.DataFrame.from_dict(data, orient='index')
        return df
    
    # Fallback to CSV if JSON not available
    csv_file = os.path.join(dataset_dir, "selected_content_statistics.csv")
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file, index_col=0)
    
    raise FileNotFoundError("Summary statistics files not found")

def visualize_rq1_community_engagement():
    """
    Visualize community engagement metrics to answer:
    RQ1: How does engagement differ across FOSS and CSPS communities?
    """
    print("Generating visualizations for RQ1: Community Engagement...")
    
    # Load data
    stats_df = load_summary_data()
    stats_df.index.name = 'dataset'
    stats_df = stats_df.reset_index()
    
    # Plot 1: Community Size Comparison
    plt.figure(figsize=(12, 8))
    community_sizes = stats_df[['dataset', 'total_posts', 'total_answers', 'total_comments']]
    community_sizes = community_sizes.melt(id_vars=['dataset'], 
                                         value_vars=['total_posts', 'total_answers', 'total_comments'],
                                         var_name='content_type', value_name='count')
    
    # Create stacked bar chart
    ax = sns.barplot(x='dataset', y='count', hue='content_type', data=community_sizes)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='center', fontsize=9)
    
    plt.title('Community Size and Composition by Platform', fontsize=14)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Content Type', title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq1_community_size.png'))
    plt.close()
    
    # Plot 2: Answer-to-Question Ratio (Responsiveness)
    plt.figure(figsize=(10, 6))
    stats_df['answer_to_post_ratio'] = stats_df['total_answers'] / stats_df['total_posts']
    stats_df['comments_per_post'] = stats_df['total_comments'] / stats_df['total_posts']
    
    ax = sns.barplot(x='dataset', y='answer_to_post_ratio', data=stats_df, palette='viridis')
    
    # Add value labels
    for i, v in enumerate(stats_df['answer_to_post_ratio']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    
    plt.title('Answer-to-Question Ratio by Platform (Responsiveness)', fontsize=14)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Ratio (Answers/Posts)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq1_answer_ratio.png'))
    plt.close()
    
    # Plot 3: Community Engagement Metrics Dashboard
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Subplot 1: Content Distribution
    ax1 = plt.subplot(gs[0, 0])
    composition_data = stats_df[['dataset', 'total_posts', 'total_answers', 'total_comments']]
    composition_data = composition_data.set_index('dataset')
    composition_data_pct = composition_data.div(composition_data.sum(axis=1), axis=0) * 100
    composition_data_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title('Content Type Distribution (%)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Percentage', fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(title='Content Type')
    
    # Subplot 2: Comments per Post
    ax2 = plt.subplot(gs[0, 1])
    sns.barplot(x='dataset', y='comments_per_post', data=stats_df, ax=ax2, palette='magma')
    ax2.set_title('Comments per Post', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylabel('Ratio', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Subplot 3: Code Snippets per Post
    ax3 = plt.subplot(gs[1, 0])
    stats_df['code_snippets_per_post'] = stats_df['num_code_snippets'].astype(float) / stats_df['total_posts']
    sns.barplot(x='dataset', y='code_snippets_per_post', data=stats_df, ax=ax3, palette='cividis')
    ax3.set_title('Code Snippets per Post (Technical Content)', fontsize=12)
    ax3.set_xlabel('Platform', fontsize=10)
    ax3.set_ylabel('Ratio', fontsize=10)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Subplot 4: Questions per Post
    ax4 = plt.subplot(gs[1, 1])
    stats_df['questions_per_post'] = stats_df['num_questions'].astype(float) / stats_df['total_posts']
    sns.barplot(x='dataset', y='questions_per_post', data=stats_df, ax=ax4, palette='rocket')
    ax4.set_title('Questions per Post (Inquiry Level)', fontsize=12)
    ax4.set_xlabel('Platform', fontsize=10)
    ax4.set_ylabel('Ratio', fontsize=10)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.suptitle('Community Engagement Metrics Dashboard', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(plots_dir, 'rq1_engagement_dashboard.png'))
    plt.close()
    
    print("RQ1 visualizations saved to:", plots_dir)

def visualize_rq2_content_characteristics():
    """
    Visualize content characteristics to answer:
    RQ2: How do content patterns differ across communities?
    """
    print("Generating visualizations for RQ2: Content Characteristics...")
    
    # Load data
    stats_df = load_summary_data()
    stats_df.index.name = 'dataset'
    stats_df = stats_df.reset_index()
    
    # Convert columns to numeric if needed
    numeric_cols = ['average_word_count', 'average_char_count', 'max_post_length', 'min_post_length']
    for col in numeric_cols:
        if stats_df[col].dtype == 'object':
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    # Plot 1: Content Length Comparison
    plt.figure(figsize=(12, 8))
    
    # Create subplot grid
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Subplot 1: Average Word Count
    ax1 = plt.subplot(gs[0, 0])
    sns.barplot(x='dataset', y='average_word_count', data=stats_df, ax=ax1, palette='viridis')
    ax1.set_title('Average Word Count', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Words', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Subplot 2: Average Character Count
    ax2 = plt.subplot(gs[0, 1])
    sns.barplot(x='dataset', y='average_char_count', data=stats_df, ax=ax2, palette='viridis')
    ax2.set_title('Average Character Count', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_ylabel('Characters', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Subplot 3: Max Post Length
    ax3 = plt.subplot(gs[1, 0])
    sns.barplot(x='dataset', y='max_post_length', data=stats_df, ax=ax3, palette='viridis')
    ax3.set_title('Maximum Post Length', fontsize=12)
    ax3.set_xlabel('Platform', fontsize=10)
    ax3.set_ylabel('Characters', fontsize=10)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Subplot 4: Min Post Length
    ax4 = plt.subplot(gs[1, 1])
    sns.barplot(x='dataset', y='min_post_length', data=stats_df, ax=ax4, palette='viridis')
    ax4.set_title('Minimum Post Length', fontsize=12)
    ax4.set_xlabel('Platform', fontsize=10)
    ax4.set_ylabel('Characters', fontsize=10)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.suptitle('Content Length Comparison Across Platforms', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(plots_dir, 'rq2_content_length.png'))
    plt.close()
    
    # Plot 2: Technical Content Indicators
    plt.figure(figsize=(14, 7))
    
    # Create data for the visualization
    tech_indicators = stats_df[['dataset', 'num_code_snippets', 'num_questions']].copy()
    tech_indicators['num_code_snippets'] = pd.to_numeric(tech_indicators['num_code_snippets'], errors='coerce')
    tech_indicators['num_questions'] = pd.to_numeric(tech_indicators['num_questions'], errors='coerce')
    
    # Normalize by total records for fair comparison
    tech_indicators['code_snippets_pct'] = tech_indicators['num_code_snippets'] / stats_df['total_records'] * 100
    tech_indicators['questions_pct'] = tech_indicators['num_questions'] / stats_df['total_records'] * 100
    
    # Prepare data for grouped bar chart
    tech_data = tech_indicators[['dataset', 'code_snippets_pct', 'questions_pct']]
    tech_data = tech_data.melt(id_vars=['dataset'], 
                           value_vars=['code_snippets_pct', 'questions_pct'],
                           var_name='indicator', value_name='percentage')
    
    # Create grouped bar chart
    ax = sns.barplot(x='dataset', y='percentage', hue='indicator', data=tech_data)
    
    # Customize chart
    plt.title('Technical Content Indicators (% of Total Content)', fontsize=14)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=45)
    
    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Code Snippets', 'Questions'], title='Indicator')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq2_technical_indicators.png'))
    plt.close()
    
    # Plot 3: Content Complexity Dashboard
    plt.figure(figsize=(14, 8))
    
    # Create a radar chart for content complexity
    # Extract metrics and normalize them for the radar chart
    metrics = ['average_word_count', 'code_snippets_per_post', 'questions_per_post', 'average_char_count']
    
    # Ensure all metrics exist and are numeric
    for metric in metrics:
        if metric not in stats_df.columns:
            if metric == 'code_snippets_per_post':
                stats_df['code_snippets_per_post'] = pd.to_numeric(stats_df['num_code_snippets'], errors='coerce') / stats_df['total_posts']
            elif metric == 'questions_per_post':
                stats_df['questions_per_post'] = pd.to_numeric(stats_df['num_questions'], errors='coerce') / stats_df['total_posts']
        stats_df[metric] = pd.to_numeric(stats_df[metric], errors='coerce')
    
    # Normalize data (0-1 scale)
    normalized_df = stats_df.copy()
    for metric in metrics:
        min_val = stats_df[metric].min()
        max_val = stats_df[metric].max()
        if max_val > min_val:  # Avoid division by zero
            normalized_df[metric] = (stats_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 0.5
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot
    ax = plt.subplot(111, polar=True)
    
    # Draw one platform at a time
    for i, platform in enumerate(normalized_df['dataset']):
        # Get values for this platform
        values = normalized_df.loc[normalized_df['dataset'] == platform, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=platform)
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Avg Word Count', 'Code Snippets\nper Post', 'Questions\nper Post', 'Avg Char Count'])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Content Complexity Comparison (Normalized)', fontsize=14, y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq2_complexity_radar.png'))
    plt.close()
    
    print("RQ2 visualizations saved to:", plots_dir)

def visualize_rq3_platform_comparison():
    """
    Visualize platform comparison to answer:
    RQ3: What are the key differences between FOSS and CSPS platforms?
    """
    print("Generating visualizations for RQ3: Platform Comparison...")
    
    # Load data
    stats_df = load_summary_data()
    stats_df.index.name = 'dataset'
    stats_df = stats_df.reset_index()
    
    # Ensure numeric columns
    for col in stats_df.columns:
        if col not in ['dataset', 'columns', 'content_type_counts']:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    # Calculate derived metrics for comparison
    stats_df['answer_ratio'] = stats_df['total_answers'] / stats_df['total_posts']
    stats_df['comment_ratio'] = stats_df['total_comments'] / stats_df['total_posts']
    stats_df['code_density'] = stats_df['num_code_snippets'] / stats_df['total_records']
    stats_df['question_density'] = stats_df['num_questions'] / stats_df['total_records']
    
    # Identify FOSS vs CSPS platforms
    # For this example, I'll classify linux, mongodb as FOSS and oracle_database, sqlite as CSPS
    # You can adjust this classification as needed
    foss_platforms = ['linux', 'mongodb', 'macos']
    csps_platforms = ['oracle_database', 'sqlite']
    
    stats_df['platform_type'] = stats_df['dataset'].apply(
        lambda x: 'FOSS' if x in foss_platforms else 'CSPS'
    )
    
    # Plot 1: FOSS vs CSPS Content Distribution
    plt.figure(figsize=(14, 8))
    
    # Aggregate data by platform type
    platform_type_data = stats_df.groupby('platform_type').agg({
        'total_posts': 'sum',
        'total_answers': 'sum',
        'total_comments': 'sum'
    })
    
    # Calculate percentages
    platform_type_data['total'] = platform_type_data.sum(axis=1)
    for col in ['total_posts', 'total_answers', 'total_comments']:
        platform_type_data[f'{col}_pct'] = platform_type_data[col] / platform_type_data['total'] * 100
    
    # Create data for plotting
    plot_data = platform_type_data[[
        'total_posts_pct', 'total_answers_pct', 'total_comments_pct'
    ]].copy()
    
    # Create stacked bar chart
    ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6), 
                    colormap='viridis', width=0.6)
    
    # Add percentage labels - THIS IS WHERE THE ERROR OCCURS
    # Fix: Get the actual values from each column to create labels
    for i, col in enumerate(plot_data.columns):
        # Get the values from this column
        values = plot_data[col].values
        
        # Get the container with the bars for this column
        container = ax.containers[i]
        
        # Create labels with the values (not the Rectangle objects)
        labels = [f"{v:.1f}%" for v in values]
        
        # Add the labels to the bars
        ax.bar_label(container, labels=labels, label_type='center', fontsize=10, color='white')
    
    plt.title('Content Type Distribution: FOSS vs CSPS', fontsize=14)
    plt.xlabel('Platform Type', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(title='Content Type', labels=['Posts', 'Answers', 'Comments'])
    plt.ylim(0, 100)  # Ensure y-axis is 0-100%
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq3_foss_vs_csps_distribution.png'))
    plt.close()
    
    # Plot 2: Key Metrics Comparison (Box plots)
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Metrics to compare
    metrics = [
        ('answer_ratio', 'Answer-to-Post Ratio'),
        ('average_word_count', 'Average Word Count'),
        ('code_density', 'Code Snippet Density'),
        ('question_density', 'Question Density')
    ]
    
    # Create a subplot for each metric
    for i, (metric, title) in enumerate(metrics):
        row, col = divmod(i, 2)
        ax = plt.subplot(gs[row, col])
        
        # Create boxplot
        sns.boxplot(x='platform_type', y=metric, data=stats_df, ax=ax)
        
        # Add individual points
        sns.stripplot(x='platform_type', y=metric, data=stats_df, 
                   ax=ax, color='black', size=4, jitter=True)
        
        # Customize plot
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Platform Type' if row == 1 else '')
        ax.set_ylabel(title)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.suptitle('Key Metrics Comparison: FOSS vs CSPS', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(plots_dir, 'rq3_metrics_comparison.png'))
    plt.close()
    
    # Plot 3: Radar Chart Comparison of Normalized Metrics
    plt.figure(figsize=(10, 10))
    
    # Select metrics for radar chart
    radar_metrics = [
        'answer_ratio', 'comment_ratio', 'average_word_count', 
        'code_density', 'question_density', 'average_char_count'
    ]
    
    # Group by platform type and calculate means
    radar_data = stats_df.groupby('platform_type')[radar_metrics].mean()
    
    # Normalize the data (0-1 scale)
    normalized_radar = radar_data.copy()
    for metric in radar_metrics:
        min_val = radar_data[metric].min()
        max_val = radar_data[metric].max()
        if max_val > min_val:
            normalized_radar[metric] = (radar_data[metric] - min_val) / (max_val - min_val)
        else:
            normalized_radar[metric] = 0.5
    
    # Number of variables
    N = len(radar_metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create labels with line breaks for better readability
    labels = [
        'Answer-to-Post\nRatio', 
        'Comment-to-Post\nRatio',
        'Average\nWord Count',
        'Code Snippet\nDensity',
        'Question\nDensity',
        'Average\nChar Count'
    ]
    
    # Create subplot
    ax = plt.subplot(111, polar=True)
    
    # Draw one platform type at a time
    for platform_type in normalized_radar.index:
        # Get values
        values = normalized_radar.loc[platform_type].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=platform_type)
        ax.fill(angles, values, alpha=0.2)
    
    # Fix axis to go in the right order and start at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Platform Type Comparison: FOSS vs CSPS', fontsize=14, y=1.1)
    
    # Add note about normalization
    plt.figtext(0.5, 0.01, "Note: All metrics are normalized for comparison", 
              ha="center", fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq3_radar_comparison.png'))
    plt.close()
    
    # Save the normalized metrics to a CSV for reference
    radar_data.to_csv(os.path.join(plots_dir, 'rq3_radar_stats.csv'))
    
    print("RQ3 visualizations saved to:", plots_dir)

def visualize_rq4_sentiment_analysis():
    """
    Visualize sentiment analysis to answer:
    RQ4: How does sentiment differ across platforms?
    - Focus specifically on Oracle Database and SQLite which have sentiment data
    """
    print("Generating visualizations for RQ4: Sentiment Analysis...")
    
    # Specify the datasets that have sentiment data
    sentiment_datasets = ['oracle_database', 'sqlite']
    print(f"Using sentiment data from: {', '.join(sentiment_datasets)}")
    
    # Load data - either from JSON or CSV
    sentiment_data = []
    
    # Try to load from the combined statistics file first
    stats_file = os.path.join(dataset_dir, "selected_content_statistics.json")
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                all_data = json.load(f)
                for dataset in sentiment_datasets:
                    if dataset in all_data:
                        data = all_data[dataset].copy()
                        data['dataset'] = dataset  # Add dataset name as a field
                        sentiment_data.append(data)
        except Exception as e:
            print(f"Error loading from combined statistics: {e}")
    
    # If that fails, try individual summary files
    if not sentiment_data:
        for dataset in sentiment_datasets:
            # Try loading from JSON first
            json_path = os.path.join(summary_dir, f"{dataset}_summary.json")
            csv_path = os.path.join(summary_dir, f"{dataset}_summary.csv")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        data['dataset'] = dataset  # Add dataset name
                        sentiment_data.append(data)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
            elif os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    # Convert to dict and add dataset name
                    row_dict = df.to_dict(orient='index')
                    if row_dict:  # If not empty
                        first_key = list(row_dict.keys())[0]
                        data = row_dict[first_key]
                        data['dataset'] = dataset
                        sentiment_data.append(data)
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
    
    if not sentiment_data:
        print("Could not load sentiment data. Skipping RQ4 visualizations.")
        return
    
    # Convert to DataFrame for analysis
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Ensure we have sentiment data to work with
    if 'sentiment' not in sentiment_df.columns:
        print("No sentiment column found in the data. Checking for alternatives...")
        # Look for any column that might contain sentiment
        sentiment_cols = [col for col in sentiment_df.columns if 'sentiment' in col.lower()]
        if sentiment_cols:
            print(f"Found potential sentiment columns: {sentiment_cols}")
            sentiment_df['sentiment'] = sentiment_df[sentiment_cols[0]]
        else:
            print("No sentiment data found. Skipping RQ4 visualizations.")
            return
    
    # Ensure sentiment is numeric
    sentiment_df['sentiment'] = pd.to_numeric(sentiment_df['sentiment'], errors='coerce')
    
    # Plot 1: Sentiment Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Create bar chart comparing sentiment across the two platforms
    ax = sns.barplot(x='dataset', y='sentiment', data=sentiment_df, palette='Blues_d')
    
    # Add data labels
    for i, v in enumerate(sentiment_df['sentiment']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=11)
    
    # Customize chart
    plt.title('Sentiment Comparison: Oracle Database vs SQLite', fontsize=14)
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    
    # Add a reference line at 0 (neutral sentiment)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral Sentiment')
    
    # Add other reference lines if helpful
    plt.axhline(y=0.05, color='gray', linestyle=':', alpha=0.3, label='Slight Positive')
    plt.axhline(y=-0.05, color='gray', linestyle=':', alpha=0.3, label='Slight Negative')
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq4_sentiment_comparison.png'))
    plt.close()
    
    # Plot 2: Sentiment vs Content Types
    plt.figure(figsize=(12, 7))
    
    # Add platform categorization
    sentiment_df['platform_type'] = sentiment_df['dataset'].apply(
        lambda x: 'CSPS'  # Both Oracle and SQLite are classified as CSPS
    )
    
    # Extract/calculate additional metrics
    for df_row in sentiment_df.iterrows():
        idx, row = df_row
        
        # Get content type counts
        if 'content_type_counts' in row:
            content_counts = row['content_type_counts']
            
            # Handle different formats
            if isinstance(content_counts, dict):
                pass  # Already in the right format
            elif isinstance(content_counts, str):
                try:
                    import ast
                    content_counts = ast.literal_eval(content_counts)
                except:
                    content_counts = {}
            
            # Calculate post ratios
            if isinstance(content_counts, dict) and 'Post' in content_counts:
                total_posts = content_counts.get('Post', 0)
                if total_posts > 0:
                    sentiment_df.at[idx, 'post_ratio'] = total_posts / sum(content_counts.values())
                    
                    # Calculate answer ratio
                    answer_count = content_counts.get('Answer', 0)
                    sentiment_df.at[idx, 'answer_ratio'] = answer_count / total_posts if total_posts > 0 else 0
                    
                    # Calculate comment ratio
                    comment_count = content_counts.get('Comment', 0)
                    sentiment_df.at[idx, 'comment_ratio'] = comment_count / total_posts if total_posts > 0 else 0
    
    # Create grid for subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot sentiment vs post types as pie charts
    for i, dataset in enumerate(sentiment_df['dataset']):
        dataset_data = sentiment_df[sentiment_df['dataset'] == dataset]
        
        # Extract content type counts
        content_counts = dataset_data['content_type_counts'].iloc[0]
        
        # Handle different formats
        if isinstance(content_counts, str):
            try:
                import ast
                content_counts = ast.literal_eval(content_counts)
            except:
                print(f"Could not parse content_type_counts for {dataset}")
                continue
                
        if not isinstance(content_counts, dict):
            print(f"Invalid content_type_counts format for {dataset}")
            continue
            
        # Extract values
        labels = list(content_counts.keys())
        sizes = list(content_counts.values())
        
        # Create pie chart
        axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   shadow=False, explode=[0.05]*len(labels))
        axes[i].set_title(f'{dataset} Content Distribution\nSentiment: {dataset_data["sentiment"].iloc[0]:.3f}')
    
    plt.suptitle('Content Distribution in Platforms with Sentiment Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq4_sentiment_content_distribution.png'))
    plt.close()
    
    # Plot 3: Content Characteristics Comparison
    # This plot will compare several content characteristics alongside sentiment
    plt.figure(figsize=(12, 8))
    
    # Ensure all necessary columns are numeric
    numeric_cols = ['average_word_count', 'average_char_count', 'num_code_snippets', 
                   'num_questions', 'total_posts', 'sentiment']
    
    for col in numeric_cols:
        if col in sentiment_df.columns:
            sentiment_df[col] = pd.to_numeric(sentiment_df[col], errors='coerce')
    
    # Calculate additional metrics
    sentiment_df['code_per_post'] = sentiment_df['num_code_snippets'] / sentiment_df['total_posts']
    sentiment_df['questions_per_post'] = sentiment_df['num_questions'] / sentiment_df['total_posts']
    
    # Create metrics for comparison
    metrics = {
        'average_word_count': 'Avg Word Count',
        'code_per_post': 'Code Snippets per Post',
        'questions_per_post': 'Questions per Post',
        'sentiment': 'Sentiment Score'
    }
    
    # Convert to long format for visualization
    plot_data = sentiment_df[['dataset'] + list(metrics.keys())].melt(
        id_vars=['dataset'], 
        value_vars=list(metrics.keys()),
        var_name='metric', 
        value_name='value'
    )
    
    # Replace metric names with readable versions
    plot_data['metric'] = plot_data['metric'].map(metrics)
    
    # Normalize the data within each metric group for better comparison
    normalized_data = plot_data.copy()
    for metric in metrics.values():
        metric_data = normalized_data[normalized_data['metric'] == metric]
        min_val = metric_data['value'].min()
        max_val = metric_data['value'].max()
        
        if max_val > min_val:  # Avoid division by zero
            normalized_data.loc[normalized_data['metric'] == metric, 'normalized_value'] = \
                (normalized_data.loc[normalized_data['metric'] == metric, 'value'] - min_val) / (max_val - min_val)
        else:
            normalized_data.loc[normalized_data['metric'] == metric, 'normalized_value'] = 0.5
    
    # Create grouped bar chart with normalized values
    g = sns.catplot(x='dataset', y='normalized_value', hue='metric', 
                  data=normalized_data, kind='bar', height=6, aspect=1.5)
    
    # Customize chart
    g.set_axis_labels("Platform", "Normalized Value (0-1)")
    g.set_xticklabels(rotation=0)
    g.fig.suptitle('Normalized Content Characteristics and Sentiment', fontsize=14, y=1.02)
    g.fig.subplots_adjust(top=0.85)
    
    # Add annotation about normalization
    plt.figtext(0.5, 0.01, 
              "Note: All metrics normalized within their ranges for comparison purposes", 
              ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rq4_normalized_metrics.png'))
    plt.close()
    
    # Save the sentiment analysis metrics to a CSV for reference
    sentiment_df.to_csv(os.path.join(plots_dir, 'rq4_sentiment_metrics.csv'))
    
    print("RQ4 visualizations saved to:", plots_dir)

def create_readme():
    """Create a README file explaining the visualizations"""
    readme_path = os.path.join(plots_dir, "README.md")
    
    with open(readme_path, 'w') as f:
        f.write("# Research Questions Visualizations\n\n")
        f.write("This directory contains visualizations generated to help answer research questions about community engagement and content patterns across different platforms.\n\n")
        
        f.write("## RQ1: Community Engagement Analysis\n\n")
        f.write("- **rq1_community_size.png**: Stacked bar chart showing the size and composition of each community (posts, answers, comments).\n")
        f.write("- **rq1_answer_ratio.png**: Bar chart showing the answer-to-post ratio for each platform (responsiveness).\n")
        f.write("- **rq1_engagement_dashboard.png**: Dashboard with multiple engagement metrics including content distribution percentage, comments per post, code snippets per post, and questions per post.\n\n")
        
        f.write("## RQ2: Content Characteristics Analysis\n\n")
        f.write("- **rq2_content_length.png**: Comparison of content length metrics across platforms, including average word count, character count, maximum and minimum post lengths.\n")
        f.write("- **rq2_technical_indicators.png**: Comparison of code snippets and questions as percentages of total content across platforms.\n")
        f.write("- **rq2_complexity_radar.png**: Radar chart showing normalized content complexity metrics across platforms, allowing for multi-dimensional comparison.\n\n")
        
        f.write("## RQ3: FOSS vs CSPS Platform Comparison\n\n")
        f.write("- **rq3_foss_vs_csps_distribution.png**: Content type distribution (posts, answers, comments) comparison between FOSS and CSPS platform categories.\n")
        f.write("- **rq3_metrics_comparison.png**: Box plots comparing key metrics between FOSS and CSPS platforms, including answer ratio, word count, code density, and question density.\n")
        f.write("- **rq3_radar_comparison.png**: Radar chart comparing normalized metrics between FOSS and CSPS platforms.\n")
        f.write("- **rq3_radar_stats.csv**: Raw data for the radar chart comparison metrics.\n\n")
        
        f.write("## RQ4: Sentiment Analysis (Oracle Database & SQLite)\n\n")
        f.write("- **rq4_sentiment_comparison.png**: Direct comparison of sentiment scores between Oracle Database and SQLite.\n")
        f.write("- **rq4_sentiment_content_distribution.png**: Pie charts showing content type distribution for each platform with sentiment data, labeled with their sentiment scores.\n")
        f.write("- **rq4_normalized_metrics.png**: Comparison of normalized content characteristics and sentiment metrics between Oracle Database and SQLite.\n")
        f.write("- **rq4_sentiment_metrics.csv**: Raw data for the sentiment analysis metrics.\n\n")
        
        f.write("## Notes\n\n")
        f.write("- The visualizations use normalized values in several charts to facilitate comparison across different scales.\n")
        f.write("- FOSS platforms include: linux, mongodb, and macos.\n") 
        f.write("- CSPS platforms include: oracle_database and sqlite.\n")
        f.write("- Sentiment data is only available for the oracle_database and sqlite datasets.\n")
    
    print(f"README created at {readme_path}")

if __name__ == "__main__":
    print("Generating visualizations for research questions...")
    visualize_rq1_community_engagement()
    visualize_rq2_content_characteristics()
    visualize_rq3_platform_comparison()
    visualize_rq4_sentiment_analysis()
    create_readme()
    print("All visualizations completed!")
