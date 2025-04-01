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
import torch
import re
from collections import defaultdict

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

def chunk_text(text, max_length=512):
    """Split text into chunks under max_length, respecting word boundaries."""
    words = text.split()
    chunks = []
    chunk = []
    total_len = 0
    for word in words:
        if total_len + len(word) + 1 <= max_length:
            chunk.append(word)
            total_len += len(word) + 1
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
            total_len = len(word) + 1
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def extract_experience_and_skills(texts):
    """
    Extracts years of experience and associated skill mentions from job descriptions.
    Returns a mapping: {years_of_experience: [skills]}.
    """
    # This model is better suited for job skill extraction (can switch to another depending on accuracy needs)
    extractor = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

    experience_skills = defaultdict(list)

    for text in texts:
        # Extract years of experience
        # e.g., "3+ years", "at least 5 years", "minimum of 2 years", etc.
        experience_matches = re.findall(r'(\d{1,2})\+?\s*(?:years|yrs)', text, re.IGNORECASE)
        years_list = [int(year) for year in experience_matches] if experience_matches else [0]

        # Truncate input to 512 tokens
        entities = []
        for chunk in chunk_text(text):
            tmp = extractor(chunk)
            entities.extend(tmp)

        # Extract words tagged as skills (entity_group might be 'SKILL' or something similar)
        skills = [e['word'] for e in entities if 'skill' in e['entity_group'].lower()]
        skills = list(set([skill.strip() for skill in skills if skill.strip()]))  # Deduplicate & clean

        for year in years_list:
            experience_skills[year].extend(skills)

    # Deduplicate and clean final output
    for year in experience_skills:
        experience_skills[year] = sorted(set(experience_skills[year]))

    return dict(experience_skills)

def main():
    df = load_data()
    experience_skill_map = extract_experience_and_skills(df['description'].dropna().tolist())
    for years, skills in sorted(experience_skill_map.items()):
        print(f"{years} years experience:")
        print(", ".join(skills))
        print("-" * 40)

if __name__ == "__main__":
    main()