from app.db.mongo import get_mongo_client
from app.utils.logger import get_logger

logger = get_logger("gold")

DB = "connectome"
CLEAN = "clean_connections"

GOLD_REGION = "gold_region_agg"
GOLD_TENSOR = "gold_tensor"


def build_region_aggregation(db):
    pipeline = [
        {
            "$group": {
                "_id": {
                    "src": "$groups.neuropil",
                    "tgt": "$groups.neuropil"
                },
                "edge_count": {"$sum": 1},
                "total_syn": {"$sum": "$counts.syn"},
                "avg_entropy": {"$avg": "$stats.metric_entropy"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "source_region": "$_id.src",
                "target_region": "$_id.tgt",
                "edge_count": 1,
                "total_syn": 1,
                "avg_entropy": 1,
            }
        },
        {"$out": GOLD_REGION}
    ]

    logger.info("Running region aggregation (Gold-A)")
    db[CLEAN].aggregate(pipeline, allowDiskUse=True)
    logger.info("Region aggregation completed")


def build_tensor(db):
    pipeline = [
        {
            "$project": {
                "src": "$groups.neuropil",
                "tgt": "$groups.neuropil",
                "metrics": {"$objectToArray": "$metrics"},
                "support": "$counts.syn"
            }
        },
        {"$unwind": "$metrics"},
        {
            "$group": {
                "_id": {
                    "src": "$src",
                    "tgt": "$tgt",
                    "metric": "$metrics.k"
                },
                "value": {"$avg": "$metrics.v"},
                "support": {"$sum": "$support"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "coords": {
                    "src": "$_id.src",
                    "tgt": "$_id.tgt",
                    "metric": "$_id.metric"
                },
                "value": 1,
                "support": 1
            }
        },
        {"$out": GOLD_TENSOR}
    ]

    logger.info("Running tensor construction (Gold-B)")
    db[CLEAN].aggregate(pipeline, allowDiskUse=True)
    logger.info("Tensor construction completed")


def main():
    client = get_mongo_client()
    db = client[DB]

    build_region_aggregation(db)
    build_tensor(db)


if __name__ == "__main__":
    main()
