from neo4j import GraphDatabase
from pymongo import MongoClient
import math

def project_tensor_to_neo4j(top_k=500):
    mongo = MongoClient("mongodb://mongo-primary:27017/?replicaSet=rs0")
    db = mongo["connectome"]
    tensor = db["gold_tensor"]

    driver = GraphDatabase.driver(
        "bolt://neo4j:7687",
        auth=("neo4j", "password")
    )

    pipeline = [
        {
            "$match": {
                "coords.src": {"$ne": None},
                "coords.tgt": {"$ne": None}
            }
        },
        {
            "$project": {
                "src": "$coords.src",
                "tgt": "$coords.tgt",
                "metric": "$coords.metric",
                "value": {
                    "$cond": [
                        {"$or": [
                            {"$eq": ["$value", None]},
                            {"$ne": ["$value", "$value"]}
                        ]},
                        0,
                        "$value"
                    ]
                },
                "support": {"$ifNull": ["$support", 1]}
            }
        },
        {
            "$group": {
                "_id": {
                    "s": "$src",
                    "t": "$tgt",
                    "m": "$metric"
                },
                "w": {
                    "$sum": {
                        "$multiply": [
                            {"$abs": "$value"},
                            {"$ln": {"$add": ["$support", 2]}}
                        ]
                    }
                },
                "n": {"$sum": 1}
            }
        },
        {
            "$match": {
                "w": {"$gt": 0}
            }
        },
        {"$sort": {"w": -1}},
        {"$limit": top_k}
    ]

    edges = list(tensor.aggregate(pipeline, allowDiskUse=True))

    if not edges:
        print("No valid tensor edges to project.")
        return

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for e in edges:
            session.run(
                """
                MERGE (a:Neuron {name:$s})
                MERGE (b:Neuron {name:$t})
                MERGE (a)-[r:SYNAPSE {metric:$m}]->(b)
                SET
                    r.weight = $w,
                    r.support = $n
                """,
                s=e["_id"]["s"],
                t=e["_id"]["t"],
                m=e["_id"]["m"],
                w=float(e["w"]),
                n=int(e["n"])
            )

    driver.close()
    print(f"Projected {len(edges)} tensor edges into Neo4j.")
