from pathlib import Path
from datetime import datetime

import pandas as pd
from pymongo import InsertOne

from app.db.mongo import get_mongo_client
from app.utils.logger import get_logger

DATA_FILE = Path("/data/proofread_connections_783.feather")
DB_NAME = "connectome"
COLLECTION = "raw_connections"
BATCH_SIZE = 50_000


logger = get_logger("raw_ingest")


def main():
    logger.info("Connecting to MongoDB (Replica Set)...")
    client = get_mongo_client()
    db = client[DB_NAME]
    col = db[COLLECTION]

    logger.info(f"Reading feather file: {DATA_FILE}")
    df = pd.read_feather(DATA_FILE)

    logger.info(f"Total rows detected: {len(df)}")
    logger.info(f"Columns detected: {list(df.columns)}")

    buffer = []
    inserted = 0
    total = len(df)

    for row in df.itertuples(index=False):
        doc = {
            "data": row._asdict(),  
            "meta": {
                "source_file": DATA_FILE.name,
                "ingest_time": datetime.utcnow()
            }
        }

        buffer.append(InsertOne(doc))

        if len(buffer) >= BATCH_SIZE:
            col.bulk_write(buffer, ordered=False)
            inserted += len(buffer)
            logger.info(f"Inserted {inserted}/{total}")
            buffer.clear()

    if buffer:
        col.bulk_write(buffer, ordered=False)
        inserted += len(buffer)

    logger.info(f"Raw ingest completed. Total inserted: {inserted}")


if __name__ == "__main__":
    main()
