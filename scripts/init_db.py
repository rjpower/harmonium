import os
import dotenv
from insight.database import *

dotenv.load_dotenv(dotenv_path=os.environ.get("SECRETS_PATH"))

if __name__ == "__main__":
    print("connecting.")
    db = DB(user=os.getenv("DB_USER"), password=os.getenv("DB_PASS"))
    print("clear")
    db.clear()
    print("setup")
    db.setup()
    print("dummy'")
    db.insert_dummy_data()
    db.close()
