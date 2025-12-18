from pymongo import MongoClient, ReadPreference

def get_mongo_client():
    uri = (
        "mongodb://"
        "mongo-primary:27017,"
        "mongo-secondary:27017/"
        "?replicaSet=rs0"
    )

    return MongoClient(
        uri,
        read_preference=ReadPreference.PRIMARY,
        socketTimeoutMS=300000,
        connectTimeoutMS=300000,
        serverSelectionTimeoutMS=300000,
    )
