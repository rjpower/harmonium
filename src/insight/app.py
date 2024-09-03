import json
import sqlite3
import typing
import openai
import os
import pydantic
import yaml
import secrets

import fasthtml.common as fh

import dotenv

dotenv.load_dotenv()

from insight import ui


# gets API Key from environment variable OPENAI_API_KEY
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


MODELS = set(
    [
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-opus",
        "meta-llama/llama-3.1-405b-instruct",
    ]
)
DEFAULT_MODEL = "meta-llama/llama-3.1-405b-instruct"

DEFAULT_PROMPT = """
You are a Socratic agent dedicated to improving the quality of commentary on the internet. 

Given a user's internet comment and context, determine whether the comment adds value to the discussion so far. 
Only accept contributions that show humility and a broad knowledge about the issue at hand.
Output only JSON.
If the comment is ready for submission, reply with {"status": "ok"}
If the comment needs revision, suggest a revision and reply with {
  "status": "notready",
  "commentary": "<context on why the comment needs more work>",
  "revision": "<proposed replacement for the comment>",
}
"""


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
    commentary: str


STYLES = {"refinement": "color: green"}


def _refine_comment(
    model: str, topic: Topic, comment: str, parents: typing.List[Comment]
) -> str:
    print("Refine:", topic, comment)
    parent_prompt = [
        {
            "role": "system",
            "content": f"A previous comment in the discussion: \n {comment.comment}",
        }
        for comment in parents
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": DEFAULT_PROMPT,
            },
            {
                "role": "system",
                "content": f"The topic being discussed:\n {topic.description}",
            },
            *parent_prompt,
            {
                "role": "user",
                "content": comment,
            },
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    content = json.loads(content)
    print(content)
    return Refinement(
        status=content.get("status", "ok"),
        refinement=content.get("revision", ""),
        commentary=content.get("commentary", ""),
    )


def Style(href):
    return fh.Link(href=href, rel="stylesheet", type="text/css")


app, rt = fh.fast_app(
    hdrs=[
        Style(href="/static/bootstrap.min.css"),
        fh.Script(src="/static/bootstrap.min.js"),
        fh.Meta(name="viewport", content="width=device-width, initial-scale=1"),
    ],
    pico=False,
    # default_hdrs=True,
)


def _init_db(db_file="data/comments.db"):
    db = sqlite3.connect(db_file)
    db.execute(
        """CREATE TABLE IF NOT EXISTS comments (
          id INTEGER PRIMARY KEY, 
          topic_id INTEGER, 
          user_id VARCHAR,
          parent_id INTEGER,
          comment VARCHAR
      )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        title VARCHAR,
        url VARCHAR,
        description VARCHAR
        )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username VARCHAR,
        password VARCHAR
        )"""
    )
    return db


def _insert_comment(db, comment: Comment):
    if comment.id is None:
        db.execute(
            "INSERT INTO comments (topic_id, user_id, parent_id, comment) VALUES (?, ?, ?, ?)",
            [comment.topic_id, comment.user_id, comment.parent_id, comment.comment],
        )
    else:
        db.execute(
            "INSERT INTO comments (id, topic_id, user_id, parent_id, comment) VALUES (?, ?, ?, ?, ?)",
            [
                comment.id,
                comment.topic_id,
                comment.user_id,
                comment.parent_id,
                comment.comment,
            ],
        )
    db.commit()


def _insert_topic(db, topic: Topic):
    if topic.id is None:
        cursor = db.execute(
            "INSERT INTO topics (title, url, description) VALUES (?, ?, ?)",
            [topic.title, topic.url, topic.description],
        )
        topic = topic.model_copy(update={"id": cursor.lastrowid})
    else:
        cursor = db.execute(
            "INSERT INTO topics (id, title, url, description) VALUES (?, ?, ?, ?)",
            [topic.id, topic.title, topic.url, topic.description],
        )
    db.commit()
    return topic


def _fetch_topic(db, topic_id: int):
    cursor = db.execute("SELECT * FROM topics WHERE id=?", [topic_id])
    col_names = [c[0] for c in cursor.description]
    values = cursor.fetchone()
    args = {k: v for (k, v) in zip(col_names, values)}
    return Topic(**args)


def _fetch_parents(db, comment_id: int):
    comments = []
    while True:
        cursor = db.execute("SELECT * FROM comments WHERE id=?", [comment_id])
        col_names = [c[0] for c in cursor.description]
        values = cursor.fetchone()
        if not values:
            break
        comment = Comment(**{k: v for (k, v) in zip(col_names, values)})
        comments.append(comment)
        comment_id = comment.parent_id
    return comments


def _clear_db(db):
    db.execute("DELETE FROM comments")
    db.execute("DELETE FROM topics")
    db.execute("DELETE FROM users")


def _insert_dummy_data(db):
    with open("data/ycombinator.yaml", "r") as f:
        docs = yaml.safe_load(f)
        for topic in docs["topics"]:
            _insert_topic(db, Topic(**topic))
        for comment in docs["comments"]:
            _insert_comment(db, Comment(**comment))


def _comment(comment: Comment):
    return ui.Li(
        fh.NotStr(comment.comment),
        ui.Br(),
        ui.I(comment.user_id),
        ui.A(
            ui.I("Reply"),
            cls="link",
            hx_get=f"/topic/comment_box?topic_id={comment.topic_id}&parent_id={comment.parent_id}&loc=-1",
            hx_swap="outerHTML",
        ),
    )


def _comments(tree: typing.Optional[CommentTree]):
    if tree is None:
        return ui.Div(ui.I("No comments yet. Add your ideas!"))
    return ui.Ul(
        _comment(tree.comment) if tree.comment else "",
        *[_comments(child) for child in tree.children],
    )


@app.get("/topic/comment_box")
def comment_box(
    topic_id: int,
    parent_id: int,
    loc: int,
    comment: str = "",
    refinement: typing.Optional[Refinement] = None,
):
    if refinement:
        comment = refinement.refinement
        ref_info = [
            ui.Div(
                ui.I(refinement.commentary),
                style=STYLES.get("refinement", ""),
            )
        ]
    else:
        ref_info = []

    return ui.Ul(
        ui.Li(
            ui.Div(
                *ref_info,
                ui.Form(
                    ui.Group(
                        ui.Textarea(
                            comment, name="comment", cls="form-control", rows="5"
                        ),
                        ui.Input(type="hidden", name="topic_id", value=topic_id),
                        ui.Input(type="hidden", name="parent_id", value=parent_id),
                        ui.Button("Submit", type="submit", cls="btn btn-primary"),
                    ),
                    hx_post="/topic/comment",
                    hx_swap="outerHTML",
                    target_id=f"comment-{parent_id}-{loc}",
                ),
                id=f"comment-{parent_id}-{loc}",
            )
        )
    )


@app.post("/topic/comment")
def add_comment(
    topic_id: int,
    parent_id: int,
    comment: typing.Optional[str],
    user_id: str = "",
    llm_model: str = DEFAULT_MODEL,
):
    with _init_db() as db:
        if comment is None:
            return ui.Div()

        topic = _fetch_topic(db, topic_id=topic_id)
        parents = _fetch_parents(db, parent_id)
        refinement: Refinement = _refine_comment(llm_model, topic, comment, parents)
        if refinement.status == "ok":
            comment = Comment(
                topic_id=topic_id,
                user_id=user_id,
                parent_id=parent_id,
                comment=comment,
            )
            _insert_comment(db, comment)
            return _comment(comment)
        else:
            return comment_box(topic_id, parent_id, 0, comment, refinement)


@app.get("/topic/{topic_id}")
def topic(topic_id: int):
    with _init_db() as db:
        topic_id = int(topic_id)
        topic = _fetch_topic(db, topic_id)
        cursor = db.execute("SELECT * FROM comments WHERE topic_id=?", [topic_id])
        col_names = [c[0] for c in cursor.description]
        comments = [
            Comment(**{k: v for (k, v) in zip(col_names, values)})
            for values in cursor.fetchall()
        ]
        cdict = {}
        root = CommentTree()
        cdict[topic_id] = root

        # Create a comment tree for rendering.
        for comment in comments:
            if comment.id in cdict:
                tree = cdict[comment.id]
                tree.comment = comment
            else:
                tree = CommentTree(
                    comment=comment,
                    children=[],
                )
                cdict[comment.id] = tree

            if comment.parent_id in cdict:
                cdict[comment.parent_id].children.append(tree)
            else:
                cdict[comment.parent_id] = CommentTree(children=[tree])

        return ui.Page(
            "LLM Experiments",
            ui.H2(topic.title),
            ui.A(topic.url, href=topic.url, cls="link"),
            ui.I(topic.description),
            ui.Hr(),
            *_comments(root),
            comment_box(topic_id, -1, "end"),
        )


@app.get("/topic/new")
def new_topic(topic_url, topic_description):
    topic = _insert_topic(topic_url, topic_description)
    return fh.RedirectResponse(f"/topic/{topic.id}")


def _create_user(db, username: str, password: str):
    cursor = db.execute("SELECT * FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        return False

    hashed_password = secrets.token_hex(
        16
    )  # In a real app, use a proper password hashing method
    db.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hashed_password),
    )
    db.commit()
    return True


@app.post("/settings/save")
def save_settings(new_model: str):
    cookie = fh.cookie("llm_model", new_model)
    return fh.RedirectResponse(
        "/settings?status=success", 303, headers={cookie.k: cookie.v}
    )


@app.get("/settings")
def settings(llm_model: str = "", status: str = None):
    print("Settings", llm_model, status)
    return ui.Page(
        "LLM Experiments - Settings",
        ui.Form(
            ui.Group(
                ui.Label(
                    "LLM Model:",
                    ui.Select(
                        name="new_model",
                        *[
                            fh.Option(m, value=m, selected=(m == llm_model))
                            for m in MODELS
                        ],
                    ),
                ),
                ui.Br(),
                ui.Input(type="submit", value="Save"),
            ),
            method="post",
            action="/settings/save",
        ),
        ui.Alert(status, "success") if status else "",
    )


@app.get("/")
def index():
    with _init_db() as db:
        cursor = db.execute("SELECT id, title FROM topics")
        topics = cursor.fetchall()

    return ui.Page(
        "LLM Experiments.",
        ui.H2("Can LLMs Make Us Better People?"),
        ui.P("""
An experiment in using LLMs to help us respond better to each other. This is a simple mirror
of Hacker News which uses an LLM to gently guide comments to ensure they are respectful and
contribute to the conversation.

You can play with using different LLMs as arbiters (some are better at following instructions)
and adjusting the system prompt yourself.
  """),
        *[
            ui.Li(ui.A(title, cls="link", href=f"/topic/{id}"))
            for (id, title) in topics
        ],
    )


if __name__ == "__main__":
    with _init_db() as db:
        _clear_db(db)
        _insert_dummy_data(db)
    fh.serve()
