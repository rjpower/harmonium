import os
import dotenv
from harmonium.database import *

if __name__ == "__main__":
    print("connecting.")
    db = DB(
        database=os.getenv("DB_NAME", None),
        db_type=os.getenv("DB_TYPE", None),
        user=os.getenv("DB_USER", None),
        password=os.getenv("DB_PASS", None),
        db_host=os.getenv("DB_HOST", None),
        db_port=os.getenv("DB_PORT", None))
    print("clear")
    db.clear()
    print("setup")
    db.setup()
    print("dummy'")
    db.insert_dummy_data()
