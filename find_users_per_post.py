# Run this on Kaggle

import pandas as pd
from google.cloud import bigquery

stackOverflow = bigquery.Client()

def getQuery(post_ids):
    post_ids_str = ', '.join(map(str, post_ids))
    query = f"""
        WITH TargetPosts AS (
            SELECT id
            FROM UNNEST([{post_ids_str}]) AS id
        )

        SELECT 
            t.id AS post_id,
            p.owner_user_id AS user_id
        FROM TargetPosts t
        LEFT JOIN `bigquery-public-data.stackoverflow.stackoverflow_posts` p
            ON t.id = p.id

        UNION ALL

        SELECT 
            t.id AS post_id,
            pa.owner_user_id AS user_id
        FROM TargetPosts t
        LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` pa
            ON t.id = pa.id

        UNION ALL

        SELECT 
            t.id AS post_id,
            c.user_id AS user_id
        FROM TargetPosts t
        LEFT JOIN `bigquery-public-data.stackoverflow.comments` c
            ON t.id = c.id;
    """
    return query

def update_csv_with_user_id(input_file, output_file):
    df = pd.read_csv(f'/kaggle/input/{input_file}')
    
    if 'post_id' not in df.columns:
        raise ValueError("CSV must have a 'post_id' column")

    post_ids = df['post_id'].dropna().unique().tolist()

    if not post_ids:
        raise ValueError("No valid 'post_id' found in CSV")

    query = getQuery(post_ids)
    response = stackOverflow.query(query).to_dataframe()

    updated_df = df.merge(response, how='left', on='post_id')

    updated_df.to_csv(f'/kaggle/working/{output_file}', index=False)
    print(f"✅ Updated CSV saved to: /kaggle/working/{output_file}")

input_file = '/red-hat/red hat enterprise linux_posts_with_comments_answers.csv'
output_file = 'updated_output.csv'
update_csv_with_user_id(input_file, output_file)