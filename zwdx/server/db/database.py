from pymongo import MongoClient
from threading import Lock

class Database:
    _instance = None
    _lock = Lock()

    def __init__(self, uri="mongodb://localhost:5561/", db_name="zwdx_db"):
        """Private constructor — use instance() to get the singleton."""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.rooms = self.db["rooms"]
        self.jobs = self.db["jobs"]
        
    @classmethod
    def instance(cls):
        """Return the singleton instance (create if it doesn't exist)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------- ROOM OPERATIONS -------------
    def add_room(self, room_data):
        """Insert a new room into the collection."""
        return self.db.rooms.insert_one(room_data).inserted_id

    def get_room(self, room_id):
        """Retrieve a room by ID."""
        from bson.objectid import ObjectId
        return self.db.rooms.find_one({"_id": ObjectId(room_id)})

    def update_room(self, room_id, update_fields):
        from bson.objectid import ObjectId
        return self.db.rooms.update_one(
            {"_id": ObjectId(room_id)},
            {"$set": update_fields}
        )

    def delete_room(self, room_id):
        from bson.objectid import ObjectId
        return self.db.rooms.delete_one({"_id": ObjectId(room_id)})

    # ------------- JOB OPERATIONS -------------
    def add_job(self, job_data):
        return self.db.jobs.insert_one(job_data).inserted_id

    def get_job(self, job_id):
        from bson.objectid import ObjectId
        return self.db.jobs.find_one({"_id": ObjectId(job_id)})

    def update_job(self, job_id, update_fields):
        from bson.objectid import ObjectId
        return self.db.jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_fields}
        )

    def delete_job(self, job_id):
        from bson.objectid import ObjectId
        return self.db.jobs.delete_one({"_id": ObjectId(job_id)})
