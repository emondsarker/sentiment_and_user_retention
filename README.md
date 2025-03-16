# Sentiment and User Retention Analysis

This project analyzes sentiment and user retention patterns in Stack Overflow communities, comparing Free and Open Source Software (FOSS) and Commercial Software and Proprietary Systems (CSPS) platforms.

## Project Structure

- `bigquery_data_collection_step_one.ipynb`: Initial data collection script that queries Stack Overflow data from BigQuery

  - Collects posts, comments, and answers for various technologies
  - Focuses on specific keywords related to FOSS and CSPS platforms
  - Exports data to CSV files for further analysis

- `bigquery_stackoverflow.ipynb`: Exploratory analysis of Stack Overflow dataset

  - Demonstrates query structure and data extraction methods
  - Used as a template/example for the main data collection

- `vader_sentiment_classifier.ipynb`: Sentiment analysis implementation

  - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
  - Processes HTML content and extracts clean text
  - Analyzes sentiment in posts, comments, and answers
  - Input: CSV files in `input_csvs/` directory
  - Output: Processed files with sentiment scores in `output_csvs/` directory

- `find_users_per_post.py`: User interaction analysis script

  - Runs on Kaggle environment
  - Maps posts to user IDs
  - Helps track user engagement and retention
  - Links posts, answers, and comments to their authors

- `extract_statistics.py`: Statistical analysis script

  - Processes dataset files and generates comprehensive statistics
  - Analyzes content patterns, post frequencies, and user engagement metrics
  - Creates summary files for each dataset
  - Handles various date formats and data cleaning

- `visualize_statistics.py`: Data visualization script
  - Creates visualizations for four research questions:
    1. Community Engagement Analysis
    2. Content Characteristics Analysis
    3. FOSS vs CSPS Platform Comparison
    4. Sentiment Analysis
  - Generates plots in the `datasets/new_rq_plots/` directory
  - Creates detailed README explaining each visualization

## Technologies Analyzed

### FOSS (Free and Open Source Software)

- Databases: MySQL, PostgreSQL, MariaDB, MongoDB, SQLite, Redis, Cassandra, CockroachDB, CouchDB, Neo4j
- Operating Systems: Linux, FreeBSD, OpenBSD, NetBSD, Android
- Virtualization: Apache CloudStack, OpenNebula, Proxmox VE, Xen Orchestra

### CSPS (Commercial Software and Proprietary Systems)

- Databases: Oracle Database, Microsoft SQL Server, IBM DB2
- Operating Systems: Microsoft Windows, macOS, iOS
- Virtualization: Hyper-V, VMware Workstation, VMware Fusion

## Data Processing Pipeline

1. **Data Collection** (`bigquery_data_collection_step_one.ipynb`)

   - Queries Stack Overflow data from BigQuery
   - Filters posts by keywords and positive scores
   - Collects related answers and comments
   - Exports to CSV files

2. **Sentiment Analysis** (`vader_sentiment_classifier.ipynb`)

   - Processes input CSV files
   - Cleans HTML content
   - Calculates sentiment scores
   - Outputs enhanced CSV files with sentiment data

3. **User Analysis** (`find_users_per_post.py`)

   - Maps content to user IDs
   - Enables user retention analysis
   - Links different types of interactions

4. **Statistical Analysis** (`extract_statistics.py`)

   - Processes datasets
   - Generates comprehensive statistics
   - Creates summary files

5. **Visualization** (`visualize_statistics.py`)
   - Creates various plots and charts
   - Analyzes different aspects of community engagement
   - Compares FOSS and CSPS platforms
   - Examines sentiment patterns

## Research Questions

1. **Community Engagement**

   - How does engagement differ across FOSS and CSPS communities?
   - Metrics: post counts, answer ratios, comment frequencies

2. **Content Characteristics**

   - How do content patterns differ across communities?
   - Analysis of post length, code snippets, question frequency

3. **Platform Comparison**

   - What are the key differences between FOSS and CSPS platforms?
   - Comparative analysis of various metrics

4. **Sentiment Analysis**
   - How does sentiment differ across platforms?
   - Focused analysis of Oracle Database and SQLite communities

## Usage

1. Data Collection:

   ```python
   jupyter notebook bigquery_data_collection_step_one.ipynb
   ```

2. Sentiment Analysis:

   ```python
   jupyter notebook vader_sentiment_classifier.ipynb
   ```

   - Place input CSV files in `input_csvs/` directory
   - Results will be in `output_csvs/` directory

3. User Analysis:

   ```python
   # Run on Kaggle environment
   python find_users_per_post.py
   ```

4. Generate Statistics:

   ```python
   python extract_statistics.py
   ```

5. Create Visualizations:
   ```python
   python visualize_statistics.py
   ```
   - Visualizations will be in `datasets/new_rq_plots/` directory

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages:
  - pandas
  - vaderSentiment
  - beautifulsoup4
  - matplotlib
  - seaborn
  - numpy
  - google-cloud-bigquery (for BigQuery access)
