import contextlib
import logging
import os
import subprocess
import sys
import time

import openai
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from harmonium.app import router
from harmonium.database import DB

def reinit_db():
    try:
        process = subprocess.Popen(
            [sys.executable, "scripts/init_db.py"],
            env=os.environ,
        )
        start_time = time.time()

        while process.poll() is None:
            if time.time() - start_time > 10:
                process.kill()
                logging.info("init_db.py execution timed out after 10 seconds")
                return
            time.sleep(0.1)

        if process.returncode == 0:
            logging.info("init_db.py executed successfully")
        else:
            logging.info(f"init_db.py failed with return code {process.returncode}")
    except Exception as e:
        logging.info(f"Error running init_db.py: {str(e)}")


def start_scheduler():
    reinit_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(reinit_db, "cron", day="*", hour=0, minute=0)
    scheduler.start()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = DB(
        database=os.getenv("DB_NAME", None),
        db_type=os.getenv("DB_TYPE", None),
        user=os.getenv("DB_USER", None),
        password=os.getenv("DB_PASS", None),
        db_host=os.getenv("DB_HOST", None),
        db_port=os.getenv("DB_PORT", None),
    )
    app.state.llm_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    start_scheduler()
    yield


def create_app(lifespan_fn=lifespan):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    app = FastAPI(lifespan=lifespan_fn)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(router)

    @app.middleware("http")
    async def _inject_dependencies(request: Request, call_next):
        request.state.db = app.state.db
        request.state.llm_client = app.state.llm_client
        response = await call_next(request)
        return response

    return app
