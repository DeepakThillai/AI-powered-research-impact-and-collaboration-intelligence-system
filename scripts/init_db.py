"""
scripts/init_db.py  —  Database initialization & demo user seeding
Run ONCE after setting up MongoDB Atlas.

Usage:
    python scripts/init_db.py           # first-time setup
    python scripts/init_db.py --reset   # drop everything and re-seed
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# Ensure directories exist before any import that triggers config
from pathlib import Path
for d in ["./data/papers", "./data/chromadb", "./data/logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)

from backend.database import DatabaseManager
from backend.auth.auth_handler import create_user, UserCreate
from loguru import logger


DEMO_USERS = [
    {
        "name": "Dr. Research Head",
        "email": "head@university.edu",
        "password": "password123",
        "department": "Computer Science",
        "role": "research_head",
    },
    {
        "name": "Dr. Alice Faculty",
        "email": "alice@university.edu",
        "password": "password123",
        "department": "Computer Science",
        "role": "faculty",
    },
    {
        "name": "Dr. Bob Faculty",
        "email": "bob@university.edu",
        "password": "password123",
        "department": "Electronics Engineering",
        "role": "faculty",
    },
    {
        "name": "Charlie Student",
        "email": "charlie@university.edu",
        "password": "password123",
        "department": "Data Science",
        "role": "student",
    },
    {
        "name": "Diana Student",
        "email": "diana@university.edu",
        "password": "password123",
        "department": "Biomedical Engineering",
        "role": "student",
    },
]


def init_database():
    print("\n" + "=" * 60)
    print("  ResearchIQ — Database Initialization")
    print("=" * 60 + "\n")

    logger.info("Connecting to MongoDB Atlas...")
    DatabaseManager.connect()
    db = DatabaseManager.get_db()

    existing_users = db.users.count_documents({})
    if existing_users > 0:
        print(f"⚠️  Database already has {existing_users} users. Skipping seed.")
        print("   Use --reset flag to wipe and re-seed.\n")
        print("📋 Existing credentials:")
        print("   Research Head : head@university.edu / password123")
        print("   Faculty       : alice@university.edu / password123")
        print("   Student       : charlie@university.edu / password123\n")
        return

    print("Creating demo users...\n")
    for user_data in DEMO_USERS:
        try:
            create_user(UserCreate(**user_data))
            print(f"  ✅  {user_data['email']:35s}  [{user_data['role']}]")
        except Exception as e:
            print(f"  ⚠️  {user_data['email']:35s}  skipped: {e}")

    print("\n" + "=" * 60)
    print("  ✅  Database initialized successfully!")
    print("=" * 60)
    print("\n📋 Demo Login Credentials:")
    print("   Research Head : head@university.edu     / password123")
    print("   Faculty       : alice@university.edu    / password123")
    print("   Faculty       : bob@university.edu      / password123")
    print("   Student       : charlie@university.edu  / password123")
    print("   Student       : diana@university.edu    / password123")
    print("\n➡️  Next step: python scripts/generate_sample_papers.py\n")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        print("\n⚠️  RESET mode — dropping all collections...\n")
        DatabaseManager.connect()
        db = DatabaseManager.get_db()
        for coll in ["users", "papers", "sessions", "analytics_cache"]:
            db[coll].drop()
            print(f"   🗑️  Dropped: {coll}")

        # Also wipe ChromaDB
        import shutil
        from backend.config import settings
        chroma_path = Path(settings.chroma_persist_path)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            print("   🗑️  Wiped ChromaDB")
        print()

    init_database()
