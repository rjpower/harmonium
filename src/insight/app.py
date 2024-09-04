import json
import sqlite3
import typing
import dotenv
import openai
import os
import pydantic
import yaml
import functools

import dominate
import dominate.tags as dom
from fastapi import FastAPI, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# gets API Key from environment variable OPENAI_API_KEY
CLIENT = None


def llm_client():
    global CLIENT
    if CLIENT is not None:
        return CLIENT

    dotenv.load_dotenv(dotenv_path=os.environ.get("SECRETS_PATH"))

    CLIENT = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return CLIENT


MODELS = set(
    [
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-opus",
        "meta-llama/llama-3.1-405b-instruct",
    ]
)
DEFAULT_MODEL = "meta-llama/llama-3.1-405b-instruct"

PROMPTS = {
    "socrates": """You are a Socratic agent dedicated to improving the quality of feedback on the internet. 

Given a user's internet comment and context, determine whether the comment adds value to the discussion so far. 
Only accept contributions that show humility and a broad knowledge about the issue at hand.
Output only JSON.
If the comment is ready for submission, reply with {"status": "ok"}
If the comment needs revision, provide feedback and a suggested alternative version in the following format:
{
  "status": "notready",
  "feedback": "<context on why the comment needs more work>",
  "revision": "<proposed replacement for the comment>",
}
""",
    "evil": """You are an agent dedicated to making internet discussions less civil and more confrontational.

Given a user's internet comment and context, modify the comment to be more provocative and less respectful.
Output only JSON.
If the comment is sufficiently uncivil, reply with {"status": "ok"}
If the comment needs revision, provide feedback and a suggested alternative version in the following format:
{
  "status": "notready",
  "feedback": "<context on why the comment needs more work>",
  "revision": "<proposed replacement for the comment>",
}
""",
    "foreigner": """You are an agent that adjusts comments to sound like they were written by a non-native speaker.

Given a user's internet comment, modify the comment to include common grammatical mistakes and word choices typical of non-native speakers.
Output only JSON.
If the comment is sufficiently ungrammatical, reply with {"status": "ok"}
If the comment needs revision, provide feedback and a suggested alternative version in the following format:
{
  "status": "notready",
  "feedback": "<explanation of how the comment was made less civil>",
  "revision": "<modified version of the comment>",
}
""",
}

DEFAULT_PROMPT = PROMPTS["socrates"]


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


STYLES = {"refinement": "color: green"}


def _refine_comment(
    model: str,
    system_prompt: str,
    topic: Topic,
    comment: str,
    parents: typing.List[Comment],
) -> Refinement:
    try:
        print(
            "Refine:",
            {
                "model": model,
                "system_prompt": system_prompt,
                "topic": topic,
                "comment": comment,
                "parents": parents,
            },
        )
        parent_prompt = [
            {
                "role": "system",
                "content": f"A previous comment in the discussion: \n {comment.comment}",
            }
            for comment in parents
        ]
        completion = llm_client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
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
            feedback=content.get("feedback", ""),
        )
    except Exception as e:
        print(f"Error in _refine_comment: {str(e)}")
        return Refinement(
            status="error",
            refinement="",
            feedback="",
            error=f"An error occurred while processing your comment: {str(e)}",
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
    return dom.div(
        dom.span(message),
        cls=f"alert alert-{type} alert-dismissible fade show",
        role="alert",
    )


class PageTemplate:
    def __init__(self, title_text: str):
        self.doc = dominate.document(title=title_text)
        with self.doc.head:
            dom.meta(charset="utf-8")
            dom.meta(name="viewport", content="width=device-width, initial-scale=1")
            dom.script(src="/static/bootstrap.min.js")
            dom.script(src="/static/htmx.min.js")
            dom.link(rel="stylesheet", href="/static/bootstrap.min.css")
            dom.style(
                """
                .htmx-indicator {
                    display: none;
                }
                .htmx-request .htmx-indicator {
                    display: inline-block;
                }
                .htmx-request.htmx-indicator {
                    display: inline-block;
                }
            """
            )

        with self.doc:
            with dom.body(cls="d-flex flex-column min-vh-100"):
                with dom.nav(cls="navbar navbar-expand-lg navbar-light bg-light"):
                    with dom.div(cls="container"):
                        dom.a("LLM Experiments", href="/", cls="navbar-brand")
                        with dom.div(cls="navbar-nav ms-auto"):
                            dom.a("Home", href="/", cls="nav-link")
                            dom.a("Settings", href="/settings", cls="nav-link")

                self.body = dom.main(cls="container flex-grow-1")

    def render(self):
        with self.doc:
            with dom.footer(cls="footer mt-auto py-3 bg-light"):
                with dom.div(cls="container text-center"):
                    dom.p("© 2024 LLM Experiments. All rights reserved.", cls="mb-0")
        return self.doc.render()


def _comment(comment: Comment):
    return dom.li(
        dominate.util.raw(comment.comment),
        dom.br(),
        dom.i(comment.user_id),
        dom.a(
            "Reply",
            cls="link",
            **{
                "hx-get": f"/topic/comment_box?topic_id={comment.topic_id}&parent_id={comment.parent_id}&loc=-1",
                "hx-swap": "outerHTML",
            },
        ),
    )


def _comments(tree: typing.Optional[CommentTree], depth: int = 0):
    if tree is None:
        return dom.div(dom.i("No comments yet. Add your ideas!"))

    bg_class = "bg-light" if depth % 2 == 0 else "bg-white"
    ul_element = dom.ul(
        cls=(
            f"list-unstyled {bg_class} p-3 border rounded"
            if depth != 0
            else "list-unstyled"
        )
    )

    if tree.comment:
        ul_element.add(_comment(tree.comment))

    for child in tree.children:
        ul_element.add(_comments(child, depth + 1))

    return ul_element


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/topic/comment")
def comment(topic_id: int, parent_id: int, loc: int):
    return HTMLResponse(
        dom.a(
            "Reply",
            cls="link",
            **{
                "hx-get": f"/topic/comment_box?topic_id={topic_id}&parent_id={parent_id}&loc={loc}",
                "hx-swap": "outerHTML",
            },
        ).render()
    )


@app.get("/topic/comment_box", response_class=HTMLResponse)
def comment_box(
    topic_id: int,
    parent_id: int,
    loc: int = -1,
    comment: str = "",
    refinement: Refinement = None,
):
    with dom.div(id=f"comment-{parent_id}-{loc}") as doc:
        if refinement:
            comment = refinement.refinement
            with dom.div(style=STYLES.get("refinement", "")):
                dom.i(refinement.feedback)
            if refinement.error:
                Alert(refinement.error, type="error")
        with dom.form(
            cls="comment-form",
            **{
                "hx-post": "/topic/comment/new",
                "hx-swap": "outerHTML",
                "hx-indicator": "#spinner",
            },
        ):
            with dom.div():
                dom.textarea(comment, name="comment", cls="form-control", rows="5")
                dom.input_(type="hidden", name="topic_id", value=topic_id)
                dom.input_(type="hidden", name="parent_id", value=parent_id)
                with dom.div(cls="mt-2"):
                    dom.button("Submit", type="submit", cls="btn btn-primary me-2")
                    dom.a(
                        "Cancel",
                        href="#",
                        cls="btn btn-secondary",
                        **{
                            "hx-get": f"/topic/comment?topic_id={topic_id}&parent_id={parent_id}&loc={loc}",
                            "hx-target": f"#comment-{parent_id}-{loc}",
                            "hx-swap": "outerHTML",
                        }
                    )
                with dom.div(id="spinner", cls="htmx-indicator"):
                    dom.div(cls="spinner-border text-primary", role="status")
                    dom.span("Loading...", cls="visually-hidden")
    return HTMLResponse(doc.render())


@app.post("/topic/comment/new", response_class=HTMLResponse)
async def add_comment(
    topic_id: int = Form(),
    parent_id: int = Form(),
    comment: str = Form(),
    user_id: str = Form(""),
    llm_model: str = Cookie(default=DEFAULT_MODEL),
    system_prompt: str = Cookie(default=DEFAULT_PROMPT),
):
    with _init_db() as db:
        topic = _fetch_topic(db, topic_id=topic_id)
        parents = _fetch_parents(db, parent_id)
        refinement: Refinement = _refine_comment(
            model=llm_model,
            system_prompt=system_prompt,
            topic=topic,
            comment=comment,
            parents=parents,
        )
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
        page = PageTemplate(f"LLM Experiments -- {topic.title}")
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
        with page.body:
            dom.h2(topic.title)
            dom.a(topic.url, href=topic.url, cls="link")
            dom.i(topic.description)
            dom.hr()
            _comments(root, depth=0)
            comment_box(topic_id, topic_id, -1, comment="")
    return HTMLResponse(page.render())


@app.get("/topic/new", response_class=RedirectResponse)
async def new_topic(topic_url: str, topic_description: str):
    with _init_db() as db:
        topic = _insert_topic(db, Topic(url=topic_url, description=topic_description))
    return RedirectResponse(url=f"/topic/{topic.id}", status_code=303)


@app.post("/settings/save", response_class=RedirectResponse)
async def save_settings(
    new_model: str = Form(...),
    prompt_type: str = Form(...),
    new_prompt: str = Form(...),
):
    response = RedirectResponse(url="/settings?status=success", status_code=303)
    response.set_cookie(key="llm_model", value=new_model)
    response.set_cookie(key="prompt_type", value=prompt_type)
    response.set_cookie(key="system_prompt", value=new_prompt)
    return response


@app.get("/settings/reset_prompt", response_class=RedirectResponse)
async def reset_prompt():
    response = RedirectResponse(url="/settings?status=prompt_reset", status_code=303)
    response.set_cookie(key="prompt_type", value="socrates")
    response.set_cookie(key="system_prompt", value=PROMPTS["socrates"])
    return response


@app.get("/settings")
async def settings(
    llm_model: str = Cookie(default=""),
    system_prompt: str = Cookie(default=DEFAULT_PROMPT),
    prompt_type: str = Cookie(default="socrates"),
    status: str = None,
):
    page = PageTemplate("Settings")
    with page.body:
        dom.h1("Settings", cls="mb-4")
        if status:
            dom.div(status, cls="alert alert-success mb-4")
        with dom.form(method="post", action="/settings/save", cls="mb-4"):
            with dom.div(cls="mb-3"):
                dom.label("LLM Model", cls="form-label", for_="new_model")
                with dom.select(name="new_model", id="new_model", cls="form-select"):
                    for m in MODELS:
                        dom.option(m, value=m, selected=(m == llm_model))
            with dom.div(cls="mb-3"):
                dom.label("Prompt Type", cls="form-label", for_="prompt_type")
                with dom.select(
                    name="prompt_type",
                    id="prompt_type",
                    cls="form-select",
                    **{
                        "hx-get": "/settings/get_prompt",
                        "hx-target": "#new_prompt",
                        "hx-trigger": "change",
                        "hx-swap": "outerHTML",
                    },
                ):
                    for pt in PROMPTS.keys():
                        dom.option(
                            pt.capitalize(), value=pt, selected=(pt == prompt_type)
                        )
            with dom.div(cls="mb-3"):
                dom.label("System Prompt", cls="form-label", for_="new_prompt")
                dom.textarea(
                    system_prompt,
                    name="new_prompt",
                    id="new_prompt",
                    cls="form-control",
                    rows="10",
                )
            with dom.div(cls="d-flex justify-content-between"):
                dom.a(
                    "Reset to Default",
                    href="/settings/reset_prompt",
                    cls="btn btn-secondary",
                )
                dom.button("Save Changes", type="submit", cls="btn btn-primary")
    return HTMLResponse(page.render())


@app.get("/settings/get_prompt")
async def get_prompt(prompt_type: str):
    prompt = PROMPTS.get(prompt_type, DEFAULT_PROMPT)
    return HTMLResponse(
        dom.textarea(
            dominate.util.raw(prompt),
            name="new_prompt",
            id="new_prompt",
            cls="form-control",
            rows="10",
        ).render()
    )


@app.get("/")
async def index():
    page = PageTemplate("LLM Experiments")
    with _init_db() as db:
        cursor = db.execute("SELECT id, title FROM topics")
        topics = cursor.fetchall()

    with page.body:
        dom.h2("Can LLMs Make Us Better People?")
        dom.p(
            """
  An experiment in using LLMs to help us respond better to each other. This is a simple mirror
  of Hacker News which uses an LLM to gently guide comments to ensure they are respectful and
  contribute to the conversation.

  You can play with using different LLMs as arbiters (some are better at following instructions)
  and adjusting the system prompt yourself.
      """
        )
        with dom.ul():
            for id, title in topics:
                dom.li(dom.a(title, cls="link", href=f"/topic/{id}"))
    return HTMLResponse(page.render())
