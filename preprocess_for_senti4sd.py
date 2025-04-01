import os
import pandas as pd

input_dir = r"C:\Users\thene\Documents\MiningData\with_user_IDs"
output_dir = r"C:\Users\thene\Documents\MiningData\preprocessed_for_senti4sd"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname in [
        "updated_actordb_posts_with_comments_answers.csv",
        "updated_alphafive_posts_with_comments_answers.csv",
        "updated_alpine_linux_posts_with_comments_answers.csv",
        "updated_amazon_aurora_posts_with_comments_answers.csv",
        "updated_ansible_posts_with_comments_answers.csv",
        "updated_arangodb_posts_with_comments_answers.csv",
        "updated_arch_linux_posts_with_comments_answers.csv",
        "updated_aws_rds_posts_with_comments_answers.csv",
        "updated_azure_sql_database_posts_with_comments_answers.csv",
        "updated_cassandra_posts_with_comments_answers.csv",
        "updated_centos_posts_with_comments_answers.csv",
        "updated_chrome_os_posts_with_comments_answers.csv",
        "updated_citusdb_posts_with_comments_answers.csv",
        "updated_cockroachdb_posts_with_comments_answers.csv",
        "updated_coldfusion_posts_with_comments_answers.csv",
        "updated_containerd_posts_with_comments_answers.csv",
        "updated_couchdb_posts_with_comments_answers.csv",
        "updated_debian_posts_with_comments_answers.csv",
        "updated_delphi_posts_with_comments_answers.csv",
        "updated_dgraph_posts_with_comments_answers.csv",
        "updated_docker_posts_with_comments_answers.csv",
        "updated_etcd_posts_with_comments_answers.csv",
        "updated_exasol_posts_with_comments_answers.csv",
        "updated_filemaker_posts_with_comments_answers.csv",
        "updated_firebird_posts_with_comments_answers.csv",
        "updated_gentoo_posts_with_comments_answers.csv",
        "updated_google_cloud_sql_posts_with_comments_answers.csv",
        "updated_hypertable_posts_with_comments_answers.csv",
        "updated_ibm_db2_posts_with_comments_answers.csv",
        "updated_influxdb_posts_with_comments_answers.csv",
        "updated_interbase_posts_with_comments_answers.csv",
        "updated_kubernetes_posts_with_comments_answers.csv",
        "updated_kvm_posts_with_comments_answers.csv",
        "updated_labview_posts_with_comments_answers.csv",
        "updated_labview_posts_with_comments_answers_1.csv"
    ]:
        print(f"⏭️ Skipping already processed: {fname}")
        continue

    if fname.endswith(".csv"):
        fpath = os.path.join(input_dir, fname)
        print(f"🔄 Processing: {fname}")

        # Force post_title to string to avoid dtype warning
        df = pd.read_csv(fpath, dtype={"post_title": str}, low_memory=False)

        df["post_title"] = df.get("post_title", "").fillna("")
        df["content_type"] = df.get("content_type", "").fillna("")
        df["content"] = df.get("content", "").fillna("")

        df["combined_text"] = df.apply(
            lambda row: f"{row['content_type']}: {row['post_title']} {row['content']}".strip(),
            axis=1
        )

        outpath = os.path.join(output_dir, fname)
        df.to_csv(outpath, index=False)
        print(f"✅ Processed: {fname}")
