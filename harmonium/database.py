import typing
from typing import List, Optional
import pydantic
import tqdm
import yaml
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, ForeignKey
from sqlalchemy.sql import select, insert
from sqlalchemy.exc import SQLAlchemyError


class User(pydantic.BaseModel):
    user_id: str
    password: str
    model: str
    prompt: str


class Topic(pydantic.BaseModel):
    id: Optional[int] = None
    url: str
    title: str
    description: str


class Comment(pydantic.BaseModel):
    id: Optional[int] = None
    topic_id: int
    user_id: str | int
    parent_id: int
    comment: str


class CommentTree(pydantic.BaseModel):
    comment: Optional[Comment] = None
    children: "List[CommentTree]" = []


class Refinement(pydantic.BaseModel):
    status: str
    refinement: str
    feedback: str
    error: Optional[str] = None


class DB:

    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        db_type: str | None = None,
        db_host: str | None = None,
        db_port: int | None = None,
    ):
        self.database = database or "harmonium"
        self.db_type = db_type or "sqlite"
        self.user = user
        self.password = password
        self.db_host = db_host or "localhost"
        self.db_port = db_port or 5432
        self.engine = self._create_engine()
        self.metadata = MetaData()
        self._define_tables()

    def _create_engine(self):
        if self.db_type == "sqlite":
            return create_engine(f"sqlite:///{self.database}")
        elif self.db_type == "postgres":
            return create_engine(f"postgresql://{self.user}:{self.password}@{self.db_host}:{self.db_port}/{self.database}")
        else:
            raise ValueError("Unsupported database type")

    def _define_tables(self):
        self.comments = Table(
            "comments",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("topic_id", Integer, ForeignKey("topics.id")),
            Column("user_id", String),
            Column("parent_id", Integer),
            Column("comment", Text),
        )

        self.topics = Table(
            "topics",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("title", String),
            Column("url", String),
            Column("description", Text),
        )

        self.users = Table(
            "users",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("username", String),
            Column("password", String),
        )

    def setup(self):
        self.metadata.create_all(self.engine)

    def clear(self):
        self.metadata.drop_all(self.engine)

    def insert_dummy_data(self):
        with open("data/ycombinator.yaml", "r") as f:
            docs = yaml.safe_load(f)
            for topic in tqdm.tqdm(docs["topics"]):
                self.insert_topic(Topic(**topic))
            for comment in tqdm.tqdm(docs["comments"]):
                self.insert_comment(Comment(**comment))

    def insert_comment(self, comment: Comment):
        with self.engine.begin() as conn:
            stmt = insert(self.comments).values(
                topic_id=comment.topic_id,
                user_id=comment.user_id,
                parent_id=comment.parent_id,
                comment=comment.comment,
            )
            if comment.id is not None:
                stmt = stmt.values(id=comment.id)
            result = conn.execute(stmt)
            comment.id = result.inserted_primary_key[0]
            return comment

    def insert_topic(self, topic: Topic):
        with self.engine.begin() as conn:
            stmt = insert(self.topics).values(
                title=topic.title,
                url=topic.url,
                description=topic.description,
            )
            if topic.id is not None:
                stmt = stmt.values(id=topic.id)
            result = conn.execute(stmt)
            topic.id = result.inserted_primary_key[0]
            return topic

    def fetch_topic(self, topic_id: int):
        with self.engine.connect() as conn:
            stmt = select(self.topics).where(self.topics.c.id == topic_id)
            cols = stmt.subquery().columns.keys()
            result = conn.execute(stmt).fetchone()
            if result:
                return Topic(**dict(zip(cols, result)))
        return None

    def fetch_comments(self, topic_id: int):
        with self.engine.connect() as conn:
            stmt = select(self.comments).where(self.comments.c.topic_id == topic_id)
            cols = stmt.subquery().columns.keys()
            results = conn.execute(stmt).fetchall()
            return [Comment(**dict(zip(cols, row))) for row in results]

    def fetch_parents(self, comment_id: int):
        comments = []
        with self.engine.connect() as conn:
            while True:
                stmt = select(self.comments).where(self.comments.c.id == comment_id)
                cols = stmt.subquery().columns.keys()
                result = conn.execute(stmt).fetchone()
                if not result:
                    break
                comment = Comment(**dict(zip(cols, result)))
                comments.append(comment)
                comment_id = comment.parent_id
        return comments

    def fetch_topics(self):
        with self.engine.connect() as conn:
            stmt = select(self.topics)
            results = conn.execute(stmt).fetchall()
            cols = stmt.subquery().columns.keys()

            print("Results:", results[0])
            return [Topic(**dict(zip(cols, row))) for row in results]
