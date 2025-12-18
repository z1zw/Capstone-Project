import math
from pymongo import InsertOne
from app.db.mongo import get_mongo_client
from app.utils.logger import get_logger
from app.clean.clean import CleanConnection

logger = get_logger("clean")

RAW_COL = "raw_connections"
CLEAN_COL = "clean_connections"
DB_NAME = "connectome"

BATCH_SIZE = 5000


def is_entity_col(name: str, value):
    return isinstance(value, int) and any(
        k in name.lower() for k in ["id", "root", "pre", "post"]
    )


def is_count_col(name: str, value):
    return isinstance(value, int) and not is_entity_col(name, value)


def is_metric_col(value):
    return isinstance(value, float)


def entropy(values):
    s = sum(values)
    if s == 0:
        return 0.0
    probs = [v / s for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def main():
    client = get_mongo_client()
    db = client[DB_NAME]

    raw = db[RAW_COL]
    clean = db[CLEAN_COL]

    sample = raw.find_one()
    if not sample:
        raise RuntimeError("Raw collection is empty")

    logger.info("Starting schema-aware clean pipeline")

    ops = []
    total = 0

    cursor = raw.find({}, batch_size=BATCH_SIZE)

    for doc in cursor:
        data = doc["data"]

        entities = {}
        counts = {}
        metrics = {}
        groups = {}

        for k, v in data.items():
            if is_entity_col(k, v):
                if "pre" in k.lower():
                    entities["source"] = v
                elif "post" in k.lower():
                    entities["target"] = v
            elif is_count_col(k, v):
                counts[k] = v
            elif is_metric_col(v):
                metrics[k] = v
            elif isinstance(v, str):
                groups[k] = v

        metric_values = list(metrics.values())
        stats = {
            "metric_max": max(metric_values) if metric_values else None,
            "metric_entropy": entropy(metric_values),
            "metric_count": len(metric_values),
        }

        clean_doc = CleanConnection(
		entities=entities,
		groups=groups,
		counts=counts,
    		metrics=metrics,
    		stats=stats,
	).model_dump()


        ops.append(InsertOne(clean_doc))
        total += 1

        if len(ops) >= BATCH_SIZE:
            clean.bulk_write(ops, ordered=False)
            logger.info(f"Cleaned {total}")
            ops.clear()

    if ops:
        clean.bulk_write(ops, ordered=False)

    logger.info(f"Clean completed. Total cleaned: {total}")


if __name__ == "__main__":
    main()
