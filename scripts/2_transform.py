import logging
import pandas as pd
import os
import serpapi
import psycopg2
from psycopg2.extras import Json
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()
from transformers import pipeline

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT")
}

# Build the DB URI from config
DB_URI = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# TEST #
engine = create_engine(DB_URI)
table_name = "job_raw"  # change this to your actual table name
df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", engine)
########

TABLE_NAME = "job_raw"
TEXT_COLUMN = "description"

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data():
    """Load job descriptions from PostgreSQL."""
    engine = create_engine(DB_URI)
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT id, {TEXT_COLUMN} FROM {TABLE_NAME}", conn)
    logging.info("Loaded %d job records", len(df))
    return df

def extract_skills(texts):
    """Use a Hugging Face model to extract keywords or named entities."""
    extractor = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    all_skills = []
    for text in texts:
        entities = extractor(text[:512])  # Truncate long text
        skills = [e["word"] for e in entities if e["entity_group"] in ["ORG", "MISC", "SKILL"]]
        all_skills.append(skills)
    return all_skills

def main():
    df = load_data()
    df["skills"] = extract_skills(df[TEXT_COLUMN].tolist())
    print(df[["id", "skills"]])

    # Optional: Save back to DB or CSV
    # df.to_sql("jobs_with_skills", engine, if_exists="replace", index=False)
    # df.to_csv("extracted_skills.csv", index=False)

if __name__ == "__main__":
    main()