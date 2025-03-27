import json
import logging
import os
import serpapi
import psycopg2
from psycopg2.extras import Json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants (consider moving to a config file or environment variables)
API_KEY = os.getenv("SERP_API_KEY")

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT")
}

SEARCH_PARAMS = {
    "engine": "google_jobs",
    "q": "transportation data engineering",
    "location": "United States",
    "api_key": API_KEY,
    "num": 1 # number of searches
}

def fetch_jobs(params):
    """Fetch jobs using SerpAPI."""
    logging.info("Fetching jobs with parameters: %s", params)
    search = serpapi.search(params)
    results = search.get_dict()
    jobs = results.get("jobs_results", [])
    logging.info("Fetched %d job(s)", len(jobs))
    return jobs


def preview_jobs(jobs):
    """Print a quick preview of job listings."""
    if not jobs:
        logging.warning("No jobs found to preview.")
        return

    logging.info("Previewing job results:")
    for i, job in enumerate(jobs):
        print(f"{i+1}. {job.get('title')} at {job.get('company_name')}")
        print(f"Location: {job.get('location')}")
        print(f"Snippet: {job.get('description')[:300]}...\n")
        print("-" * 80)


def load_jobs_to_postgres(jobs, db_config):
    """Load job listings into PostgreSQL database."""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        logging.info("Connected to PostgreSQL")

        for job in jobs:
            cursor.execute("""
                INSERT INTO job_raw (
                    job_id, job_title, company, location, source, share_link,
                    tags, posted_at, salary, description,
                    highlights, apply_options, raw_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                job.get("job_id"),
                job.get("title"),
                job.get("company_name"),
                job.get("location"),
                job.get("via"),
                job.get("share_link"),
                job.get("extensions"),
                job.get("detected_extensions", {}).get("posted_at"),
                job.get("detected_extensions", {}).get("salary"),
                job.get("description"),
                Json(job.get("job_highlights")),
                Json(job.get("apply_options")),
                Json(job)
            ))

        conn.commit()
        logging.info("Jobs successfully inserted into the database.")

    except Exception as e:
        logging.error("Error loading jobs into database: %s", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logging.info("Database connection closed.")


def test_db_connection(config):
    """Test PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**config)
        logging.info("✅ Connected to the database successfully!")
        conn.close()
    except Exception as e:
        logging.error("❌ Database connection failed: %s", e)


def main():
    test_db_connection(DB_CONFIG)
    jobs = fetch_jobs(SEARCH_PARAMS)
    preview_jobs(jobs)
    if jobs:
        load_jobs_to_postgres(jobs, DB_CONFIG)


if __name__ == "__main__":
    main()