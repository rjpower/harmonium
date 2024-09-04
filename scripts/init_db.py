from insight.app import _init_db, _clear_db, _insert_dummy_data

if __name__ == "__main__":
    with _init_db() as db:
        _clear_db(db)
        _insert_dummy_data(db)