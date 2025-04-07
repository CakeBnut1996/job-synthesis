import logging
import pandas as pd
import os, sys
import serpapi
import psycopg2
from psycopg2.extras import Json
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv() # need to load whenever add new config to the file
from transformers import pipeline
import torch
import re
from collections import defaultdict

out_dir = os.getenv("OUTPUT_DIR")
if not out_dir or not os.path.isdir(out_dir):
    sys.exit(f"ERROR: OUTPUT_DIR is not set or does not exist. Please check your environment variable or create the folder manually.\nReceived: {out_dir}")

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

# Optional: a simple chunker to avoid 1024-token limit
def chunk_text(text, max_tokens=512):
    return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

# Helper to extract years of experience
def extract_years(text):
    matches = re.findall(r'(\d{1,2})\+?\s*(?:years|yrs)', text, re.IGNORECASE)
    return [int(m) for m in matches] if matches else [0]

# Main processing function
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_extractor = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", grouped_entities=True)

def summarize_experience_to_skills(descriptions):
    experience_skills = defaultdict(list)

    for text in descriptions:
        years_list = extract_years(text)

        # Summarize long descriptions first
        summary_chunks = []
        for chunk in chunk_text(text):
            try:
                summary = summarizer(chunk, min_length=5, max_length=30)[0]['summary_text']
                summary_chunks.append(summary)
            except:
                continue  # In case of token length or model issues

        combined_summary = " ".join(summary_chunks)

        # Extract entities (skills) from summary
        ner_entities = ner_extractor(combined_summary)
        skills = [ent['word'] for ent in ner_entities if 'qualifications' in ent['entity_group'].lower() or 'MISC' in ent['entity_group']]
        skills = list(set(s.strip() for s in skills if s.strip()))

        # Map to years of experience
        for y in years_list:
            experience_skills[y].extend(skills)

    # Final cleanup
    for year in experience_skills:
        experience_skills[year] = sorted(set(experience_skills[year]))

    return dict(experience_skills)

def main():
    df = load_data()
    descriptions = df['description'].dropna().tolist()
    exp_skill_map = summarize_experience_to_skills(descriptions)
    exp_data = []
    for yrs, skills in sorted(exp_skill_map.items()):
        exp_data.append({'Years of Experience': yrs, 'Skills': ", ".join(skills)})

    df_skills = pd.DataFrame(exp_data)
    df_skills.to_csv(os.path.join(out_dir, "experience_skills.csv"), index=False)


if __name__ == "__main__":
    main()