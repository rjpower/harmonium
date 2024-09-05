import typing
import pydantic
import tqdm
import yaml
import psycopg2


class User(pydantic.BaseModel):
    user_id: str
    password: str
    model: str
    prompt: str


class Topic(pydantic.BaseModel):
    id: typing.Optional[int] = None
    url: str
    title: str
    description: str


class Comment(pydantic.BaseModel):
    id: typing.Optional[int] = None
    topic_id: int
    user_id: str | int
    parent_id: int
    comment: str


class CommentTree(pydantic.BaseModel):
    comment: typing.Optional[Comment] = None
    children: "typing.List[CommentTree]" = []


class Refinement(pydantic.BaseModel):
    status: str
    refinement: str
    feedback: str
    error: typing.Optional[str] = None


class DB:

    def __init__(
        self,
        user: str,
        password: str,
        db_host: str = "localhost",
        db_port: int = 5432,
        database: str = "harmonium",
    ):
        self.conn = psycopg2.connect(
            host=db_host, port=db_port, database=database, user=user, password=password
        )

    def close(self):
        if self.conn:
            self.conn.close()

    def setup(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS comments (
                    id SERIAL PRIMARY KEY,
                    topic_id INTEGER,
                    user_id VARCHAR,
                    parent_id INTEGER,
                    comment TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR,
                    url VARCHAR,
                    description TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR,
                    password VARCHAR
                )
                """
            )
        self.conn.commit()

    def clear(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM comments")
            cur.execute("DELETE FROM topics")
            cur.execute("DELETE FROM users")
        self.conn.commit()

    def insert_dummy_data(self):
        with open("data/ycombinator.yaml", "r") as f:
            docs = yaml.safe_load(f)
            for topic in tqdm.tqdm(docs["topics"]):
                self.insert_topic(Topic(**topic))
            for comment in tqdm.tqdm(docs["comments"]):
                self.insert_comment(Comment(**comment))

    def insert_comment(self, comment: Comment):
        with self.conn.cursor() as cur:
            if comment.id is None:
                cur.execute(
                    """
                    INSERT INTO comments (topic_id, user_id, parent_id, comment)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        comment.topic_id,
                        comment.user_id,
                        comment.parent_id,
                        comment.comment,
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO comments (id, topic_id, user_id, parent_id, comment)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        comment.id,
                        comment.topic_id,
                        comment.user_id,
                        comment.parent_id,
                        comment.comment,
                    ),
                )
        self.conn.commit()

    def insert_topic(self, topic: Topic):
        with self.conn.cursor() as cur:
            if topic.id is None:
                cur.execute(
                    """
                    INSERT INTO topics (title, url, description)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (topic.title, topic.url, topic.description),
                )
                topic_id = cur.fetchone()[0]
                topic = topic.model_copy(update={"id": topic_id})
            else:
                cur.execute(
                    """
                    INSERT INTO topics (id, title, url, description)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (topic.id, topic.title, topic.url, topic.description),
                )
        self.conn.commit()
        return topic

    def fetch_topic(self, topic_id: int):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM topics WHERE id = %s", (topic_id,))
            col_names = [desc[0] for desc in cur.description]
            values = cur.fetchone()
            if values:
                args = dict(zip(col_names, values))
                return Topic(**args)
        return None

    def fetch_comments(self, topic_id: int):
        comments = []
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM comments WHERE topic_id = %s", (topic_id,))
            col_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            for row in rows:
                comments.append(Comment(**dict(zip(col_names, row))))
        return comments

    def fetch_parents(self, comment_id: int):
        comments = []
        with self.conn.cursor() as cur:
            while True:
                cur.execute("SELECT * FROM comments WHERE id = %s", (comment_id,))
                col_names = [desc[0] for desc in cur.description]
                values = cur.fetchone()
                if not values:
                    break
                comment = Comment(**dict(zip(col_names, values)))
                comments.append(comment)
                comment_id = comment.parent_id
        return comments

    def fetch_topics(self):
        topics = []
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM topics")
            col_names = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                topics.append(Topic(**dict(zip(col_names, row))))
        return topics
