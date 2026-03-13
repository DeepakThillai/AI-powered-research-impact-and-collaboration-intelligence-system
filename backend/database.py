"""
database.py - MongoDB Atlas connection and collection management
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from loguru import logger
from backend.config import settings


class DatabaseManager:
    _client: MongoClient = None
    _db: Database = None

    @classmethod
    def connect(cls):
        """Initialize MongoDB Atlas connection."""
        if not settings.mongodb_url:
            raise ValueError(
                "MONGODB_URL is not set in your .env file!\n"
                "It should look like:\n"
                "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/"
                "?retryWrites=true&w=majority&appName=Cluster0"
            )
        try:
            # Detect Atlas SRV vs plain connection string
            is_atlas = "mongodb+srv://" in settings.mongodb_url or "mongodb.net" in settings.mongodb_url
            connect_kwargs = {
                "serverSelectionTimeoutMS": 10000,
                "connectTimeoutMS": 10000,
                "socketTimeoutMS": 30000,
            }
            if is_atlas:
                connect_kwargs["tls"] = True
                connect_kwargs["tlsAllowInvalidCertificates"] = False

            cls._client = MongoClient(settings.mongodb_url, **connect_kwargs)
            cls._client.server_info()  # Force connection check
            cls._db = cls._client[settings.mongodb_db_name]
            cls._create_indexes()
            logger.info(f"✅ Connected to MongoDB: {settings.mongodb_db_name}")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise

    @classmethod
    def _create_indexes(cls):
        """Create indexes safely (ignore if already exist)."""
        try:
            cls._db.users.create_index([("email", ASCENDING)], unique=True)
            cls._db.users.create_index([("user_id", ASCENDING)], unique=True)
            cls._db.papers.create_index([("paper_id", ASCENDING)], unique=True)
            cls._db.papers.create_index([("department", ASCENDING)])
            cls._db.papers.create_index([("publication_year", DESCENDING)])
            cls._db.papers.create_index([("uploaded_by", ASCENDING)])
            cls._db.papers.create_index([
                ("title", "text"), ("abstract", "text"), ("keywords", "text")
            ])
            cls._db.sessions.create_index([("token", ASCENDING)])
            cls._db.sessions.create_index(
                [("expires_at", ASCENDING)], expireAfterSeconds=0
            )
            logger.info("✅ Database indexes ensured")
        except Exception as e:
            logger.warning(f"Index creation warning (non-fatal): {e}")

    @classmethod
    def get_db(cls) -> Database:
        if cls._db is None:
            cls.connect()
        return cls._db

    @classmethod
    def get_collection(cls, name: str) -> Collection:
        return cls.get_db()[name]

    @classmethod
    def disconnect(cls):
        if cls._client:
            cls._client.close()
            logger.info("MongoDB connection closed")


def get_users_collection() -> Collection:
    return DatabaseManager.get_collection("users")

def get_papers_collection() -> Collection:
    return DatabaseManager.get_collection("papers")

def get_sessions_collection() -> Collection:
    return DatabaseManager.get_collection("sessions")

def get_analytics_collection() -> Collection:
    return DatabaseManager.get_collection("analytics_cache")
