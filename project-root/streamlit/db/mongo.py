from pymongo import MongoClient, ReadPreference

def get_db():
    uri = (
        "mongodb://"
        "mongo-primary:27017,"
        "mongo-secondary:27017/"
        "?replicaSet=rs0"
    )

    client = MongoClient(
        uri,
        read_preference=ReadPreference.PRIMARY_PREFERRED,
        serverSelectionTimeoutMS=300000,
        socketTimeoutMS=300000,
    )

    return client["connectome"]
