import json
import sqlite3
import typing
import openai
import os
import pydantic
import yaml
import secrets
import dominate
from dominate.tags import *
from fastapi import FastAPI, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import dotenv

dotenv.load_dotenv()

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


def Alert(message: str, type: str):
    return div(
        span(message),
        cls=f"alert alert-{type} alert-dismissible fade show",
        role="alert",
    )


def Ul(*args, **kw):
    return ul(*args, **kw)


def Li(*args, **kw):
    return li(*args, **kw)


def Page(title_text: str):
    doc = dominate.document(title(title_text))
    with doc:
        nav(
            div(
                ul(
                    li(a("Home", href="/", cls="nav-link"), cls="nav-item"),
                    li(a("Settings", href="/settings", cls="nav-link"), cls="nav-item"),
                    cls="navbar-nav me-auto mb-2 mb-lg-0",
                ),
            ),
            cls="navbar navbar-expand-lg bg-body-tertiary navbar-fixed-top",
        )

    with doc.head:
        script(src="/static/bootstrap.min.js")
        script(src="/static/htmx.min.js")
        link(rel="stylesheet", href="/static/bootstrap.min.css")
    return doc


def _comment(comment: Comment):
    return li(
        comment.comment,
        br(),
        i(comment.user_id),
        a(
            "Reply",
            cls="link",
            **{
                "hx-get": f"/topic/comment_box?topic_id={comment.topic_id}&parent_id={comment.parent_id}&loc=-1",
                "hx-swap": "outerHTML",
            },
        ),
    )


def _comments(tree: typing.Optional[CommentTree]):
    if tree is None:
        return div(i("No comments yet. Add your ideas!"))
    ul_element = ul()
    if tree.comment:
        ul_element.add(_comment(tree.comment))
    for child in tree.children:
        ul_element.add(_comments(child))
    return ul_element


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/topic/comment_box", response_class=HTMLResponse)
def comment_box(
    topic_id: int,
    parent_id: int,
    loc: int = -1,
    comment: str = "",
    refinement: Refinement = None,
):
    with div(id=f"comment-{parent_id}-{loc}") as doc:
        if refinement:
            comment = refinement.refinement
            with div(style=STYLES.get("refinement", "")):
                i(refinement.commentary)
        with form(
            cls="comment-form",
            **{
                "hx-post": "/topic/comment/new",
                "hx-swap": "outerHTML",
            },
        ):
            with div():
                textarea(comment, name="comment", cls="form-control", rows="5")
                input_(type="hidden", name="topic_id", value=topic_id)
                input_(type="hidden", name="parent_id", value=parent_id)
                button("Submit", type="submit", cls="btn btn-primary")
    return HTMLResponse(doc.render())


@app.post("/topic/comment/new", response_class=HTMLResponse)
async def add_comment(
    topic_id: int = Form(),
    parent_id: int = Form(),
    comment: str = Form(),
    user_id: str = Form(""),
    llm_model: str = Cookie(default=DEFAULT_MODEL),
):
    with _init_db() as db:
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
            return HTMLResponse(_comment(comment).render())
        else:
            return comment_box(topic_id, parent_id, -1, comment, refinement=refinement)


@app.get("/topic/{topic_id}", response_class=HTMLResponse)
async def topic(topic_id: int):
    with _init_db() as db:
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

        doc = Page("LLM Experiments")
        with doc:
            h2(topic.title)
            a(topic.url, href=topic.url, cls="link")
            i(topic.description)
            hr()
            _comments(root)
            comment_box(topic_id, topic_id, -1, comment="")
        return HTMLResponse(doc.render())


@app.get("/topic/new", response_class=RedirectResponse)
async def new_topic(topic_url: str, topic_description: str):
    with _init_db() as db:
        topic = _insert_topic(db, Topic(url=topic_url, description=topic_description))
    return RedirectResponse(url=f"/topic/{topic.id}", status_code=303)


@app.post("/settings/save", response_class=RedirectResponse)
async def save_settings(new_model: str = Form(...)):
    response = RedirectResponse(url="/settings?status=success", status_code=303)
    response.set_cookie(key="llm_model", value=new_model)
    return response


@app.get("/settings", response_class=HTMLResponse)
async def settings(llm_model: str = Cookie(default=""), status: str = None):
    doc = Page("LLM Experiments - Settings")
    with doc:
        with form(method="post", action="/settings/save"):
            with div():
                label("LLM Model:")
                with select(name="new_model"):
                    for m in MODELS:
                        option(m, value=m, selected=(m == llm_model))
                br()
                input_(type="submit", value="Save")
        if status:
            div(status, cls="alert alert-success")
    return HTMLResponse(doc.render())


@app.get("/", response_class=HTMLResponse)
async def index():
    with _init_db() as db:
        cursor = db.execute("SELECT id, title FROM topics")
        topics = cursor.fetchall()

    doc = Page("LLM Experiments")
    with doc:
        h2("Can LLMs Make Us Better People?")
        p("""
An experiment in using LLMs to help us respond better to each other. This is a simple mirror
of Hacker News which uses an LLM to gently guide comments to ensure they are respectful and
contribute to the conversation.

You can play with using different LLMs as arbiters (some are better at following instructions)
and adjusting the system prompt yourself.
  """)
        with ul():
            for id, title in topics:
                li(a(title, cls="link", href=f"/topic/{id}"))
    return HTMLResponse(doc.render())


if __name__ == "__main__":
    with _init_db() as db:
        _clear_db(db)
        _insert_dummy_data(db)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
