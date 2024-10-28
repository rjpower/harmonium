import contextlib
import json
import logging
import os
import re
import typing

import dominate
import dominate.tags as dom
import openai
import json
import yaml
from fastapi import Cookie, FastAPI, Form, Response, Request, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from harmonium.database import User, Comment, Topic, DB, CommentTree, Refinement


MODELS = set(
    [
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-opus",
        "meta-llama/llama-3.1-405b-instruct",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
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


def refine_comment(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    topic: Topic,
    comment: str,
    parents: typing.List[Comment],
) -> Refinement:
    try:
        logging.info(
            "Refine: %s",
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
        completion = client.chat.completions.create(
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
        logging.info("Refinment done %s", completion)
        content = completion.choices[0].message.content
        content = json.loads(content)
        result = Refinement(
            status=content.get("status", "ok"),
            refinement=content.get("revision", ""),
            feedback=content.get("feedback", ""),
        )
        return result
    except Exception as e:
        return Refinement(
            status="error",
            refinement="",
            feedback="",
            error=f"An error occurred while processing your comment: {str(e)}",
        )


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
.comment-thread {
  border-left: 2px solid #007bff;
  padding-left: 15px;
  margin-left: 15px;
}

.comment {
  margin-bottom: 15px;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}

.commenter-name {
  font-weight: bold;
  color: #007bff;
  margin-bottom: 5px;
}

.refinement {
   color: #28a745;
   font-style: italic;
   margin-top: 5px;
}

htmx-indicator {
   display: none;
}

htmx-request .htmx-indicator {
   display: inline-block;
}

htmx-request.htmx-indicator {
   display: inline-block;
}

.navbar-nav .nav-link img {
    filter: brightness(0.6);
    transition: filter 0.3s ease;
}

.navbar-nav .nav-link:hover img {
    filter: brightness(1);
}
"""
            )

        with self.doc:
            with dom.body(cls="d-flex flex-column min-vh-100"):
                with dom.nav(cls="navbar navbar-expand-lg navbar-light bg-light"):
                    with dom.div(cls="container"):
                        dom.a("Harmonium", href="/", cls="navbar-brand")
                        with dom.div(cls="navbar-nav ms-auto"):
                            dom.a("Conversation Demo", href="/demo", cls="nav-link")
                            dom.a("Settings", href="/settings", cls="nav-link")
                            dom.a(
                                dom.img(src="static/github-mark.svg", alt="GitHub", width="24", height="24", cls="me-1"),
                                "GitHub",
                                href="https://github.com/rjpower/harmonium",
                                cls="nav-link d-flex align-items-center",
                                target="_blank"
                            )

                self.body = dom.main(cls="container flex-grow-1")

    def render(self):
        with self.doc:
            with dom.footer(cls="footer mt-auto py-3 bg-light"):
                with dom.div(cls="container text-center"):
                    dom.p("© 2024 All rights reserved.", cls="mb-0")
        return self.doc.render()


def _comment(comment: Comment):
    return dom.li(
        dominate.util.raw(hacker_news_to_html(comment.comment)),
        dom.br(),
        dom.i(f"-- {comment.user_id}"),
        dom.a(
            "Reply",
            cls="link",
            **{
                "hx-get": f"/topic/comment_box?topic_id={comment.topic_id}&parent_id={comment.parent_id}&loc=-1",
                "hx-swap": "outerHTML",
            },
        ),
        cls="comment",
    )


def _comments(tree: typing.Optional[CommentTree], depth: int = 0):
    if tree is None:
        return dom.div(dom.i("No comments yet. Add your ideas!"))

    ul_element = dom.ul(cls="list-unstyled " + "comment-thread" if depth > 0 else "")

    if tree.comment:
        ul_element.add(_comment(tree.comment))

    for child in tree.children:
        ul_element.add(_comments(child, depth + 1))

    return ul_element

router = APIRouter()


@router.get("/topic/comment")
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


@router.get("/topic/comment_box", response_class=HTMLResponse)
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
            with dom.div(cls="refinement"):
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
                        },
                    )
                with dom.div(id="spinner", cls="htmx-indicator"):
                    dom.div(cls="spinner-border text-primary", role="status")
                    dom.span("Loading...", cls="visually-hidden")
    return HTMLResponse(doc.render())


@router.post("/topic/comment/new", response_class=HTMLResponse)
async def add_comment(
    request: Request,
    topic_id: int = Form(),
    parent_id: int = Form(),
    comment: str = Form(),
    user_id: str = Form(""),
    llm_model: str = Cookie(default=DEFAULT_MODEL),
    prompt_type: str = Cookie(default="socrates"),
):
    db = request.state.db
    topic = db.fetch_topic(topic_id=topic_id)
    parents = db.fetch_parents(parent_id)
    refinement: Refinement = refine_comment(
        client=request.state.llm_client,
        model=llm_model,
        system_prompt=PROMPTS.get(prompt_type, DEFAULT_PROMPT),
        topic=topic,
        comment=comment,
        parents=parents,
    )
    if refinement.status == "ok":
        new_comment = Comment(
            topic_id=topic_id,
            user_id=user_id,
            parent_id=parent_id,
            comment=comment,
        )
        comment = db.insert_comment(new_comment)
        return HTMLResponse(_comment(comment).render())
    else:
        return comment_box(topic_id, parent_id, -1, comment, refinement=refinement)


def hacker_news_to_html(text):
    """Converts a Hacker News formatted string into HTML."""
    text = re.sub(r"\|(https?://\S+)", r'</a><a href="\1">\1', text)
    text = text.replace("\n\n", "</p><p>")
    text = text.replace("\n", "<br>")
    text = re.sub(r"^- ", r"<li>", text)
    text = re.sub(r"(?<!</li>)\n(?=- )", r"</li>\n", text)
    text = re.sub(r"\[(\d+)\]", r"<sup>[\1]</sup>", text)
    text = text.replace("--------------", "<hr>")
    text = text.replace("&#x27;", "'")
    return text


def _build_tree(comments: typing.List[Comment], topic_id: int):
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
    return root


@router.get("/topic/{topic_id}", response_class=HTMLResponse)
async def topic(request: Request, topic_id: int):
    db = request.state.db
    topic = db.fetch_topic(topic_id)
    comments = db.fetch_comments(topic_id)

    page = PageTemplate(f"LLM Experiments -- {topic.title}")
    comment_tree = _build_tree(comments, topic_id)

    with page.body:
        dom.h2(topic.title)
        if topic.url:
            dom.a(topic.url, href=topic.url, cls="link")
            dom.hr()
        if topic.description:
            dom.i(dominate.util.raw(hacker_news_to_html(topic.description)))
            dom.hr()
        _comments(comment_tree, depth=0)
        comment_box(topic_id, topic_id, -1, comment="")
    return HTMLResponse(page.render())


@router.post("/topic/new", response_class=RedirectResponse)
async def new_topic(
    request: Request,
    topic_url: str = Form(),
    title: str = Form(),
    topic_description: str = Form(),
):
    db = request.state.db
    topic = db.insert_topic(
        Topic(
            topic_id=None, title=title, url=topic_url, description=topic_description
        ),
    )
    return RedirectResponse(url=f"/topic/{topic.id}", status_code=303)


@router.post("/settings/save", response_class=RedirectResponse)
async def save_settings(
    new_model: str = Form(...),
    prompt_type: str = Form(...),
):
    response = RedirectResponse(url="/settings?status=success", status_code=303)
    response.set_cookie(key="llm_model", value=new_model)
    response.set_cookie(key="prompt_type", value=prompt_type)
    return response


@router.get("/settings/reset_prompt", response_class=RedirectResponse)
async def reset_prompt():
    response = RedirectResponse(url="/settings?status=prompt_reset", status_code=303)
    response.set_cookie(key="prompt_type", value="socrates")
    response.set_cookie(key="system_prompt", value=PROMPTS["socrates"])
    return response


@router.get("/settings")
async def settings(
    llm_model: str = Cookie(default=""),
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
                    name="prompt_type", id="prompt_type", cls="form-select"
                ):
                    for pt in PROMPTS.keys():
                        dom.option(
                            pt.capitalize(), value=pt, selected=(pt == prompt_type)
                        )
            with dom.div(cls="d-flex justify-content-between"):
                dom.a(
                    "Reset to Default",
                    href="/settings/reset_prompt",
                    cls="btn btn-secondary",
                )
                dom.button("Save Changes", type="submit", cls="btn btn-primary")
    return HTMLResponse(page.render())


@router.get("/")
async def index(request: Request):
    page = PageTemplate("LLM Experiments")
    db = request.state.db
    topics = db.fetch_topics()

    with page.body:
        dom.h2("Better Conversations through ML?")
        dominate.util.raw(
            """
  <p>
  Comments on the internet. Like everyone else, I can't help to look at them, only to quickly want to look away.

  <p>
  Somehow no matter the audience or dryness of the topic, comment forums seem to almost 
  inevitably spiral into a chaotic mess of personal attacks. And yet, there's <i>so much 
  value</i> to be extracted (...much of the time) from the knowledge and 
  perspective of others.

  <p> I was curious if we could use LLMs (and how much it would cost) to let us communicate with
  each other better. Instead of filtering, what if we let LLMs help us express ourselves
  to each other in a healthier manner?

  <p> Unsurprisingly, the LLMs are good at deflecting obviously personal
  attacks.  Perhaps more surprising was that they aren't that easy to jailbreak
  out of their system prompt, and that LLama3-405B was the best of all of models
  I tried at following prompts. All of the models tend to rewrite
  comments in a relatively stilted manner (sigh, other than of course when you
  turn on the "evil" prompt), but it's not bad.
  
  <p>To see an example of how LLMs handle a charged conversation, see the <a
  href="/demo">conversation demo</a> page.  
      """
        )
        dom.h2("Cost")
        dom.p(
            """
So how much does it cost to do this? It's surprisingly affordable, and I imagine
someone will figure out how to package it into an API soon enough. Processing
the average comment and parent tree costs about $0.002 (2/10 of a cent) when 
using a hosted LLama3 provider."""
        )

        dom.p(
            """
With some simple filters and cheaper models in front of it, it would certainly
be viable to deploy for most use cases. (I mean, if you've got enough users that
this matters, you probably can afford it...). A risk would be someone
figuring out how to jailbreak and use you as a free LLM service, but rate limits
and filters would likely catch the vast majority of freeloaders.
"""
        )

        dom.h2("Example Site")
        dominate.util.raw("""
        <p>
  Below I mocked up a simple replica of <a
  href="https://news.ycombinator.com">Hacker News</a> which uses an LLM to
  gently guide comments to ensure they are respectful and contribute to the
  conversation.  Feel free to try adding comments and interacting with them
  (topics and comments are reset daily).
  <p>
You can also change the system prompt to make the LLM deliberately evil or
  incoherent if you like in the <a href="settings">settings</a>.
  """)
        with dom.ol():
            for topic in topics:
                dom.li(dom.a(topic.title, cls="link", href=f"/topic/{topic.id}"))
    return HTMLResponse(page.render())


@router.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    with open("data/comment_demo.yaml", "r") as file:
        data = yaml.safe_load(file)

    models = set()
    for item in data:
        models.update(item['refinement'].keys())

    page = PageTemplate("Model Comparison")

    with page.doc.head:
        dom.style("""
            select { margin-bottom: 20px; }
            .comment-pair { display: flex; margin-bottom: 20px; }
            .original, .refined { flex: 1; padding: 10px; margin: 5px; border: 1px solid #ccc; }
            .original { background-color: #ffeeee; }
            .refined { background-color: #eeffee; }
            ins { background-color: #aaffaa; text-decoration: none; }
            del { background-color: #ffaaaa; text-decoration: line-through; }
        """)

    with page.body:
        dom.h1("Model Comment Rewrite Comparison")
        with dom.select(id="modelSelect", onchange="updateComments()"):
            for model in models:
                dom.option(model, value=model)

        dom.p()
        dom.span(
            "How do different models handle a charged conversation? Below is a converstation from"
        )
        dom.a("LWN", href="https://lwn.net/Articles/986528/")
        dom.span(" that went off the rails.")
        dom.div(id="comments")

        dom.script(
            dominate.util.raw(f"""
            const data = {json.dumps(data)};
            function updateComments() {{
                const model = document.getElementById('modelSelect').value;
                const commentsDiv = document.getElementById('comments');
                commentsDiv.innerHTML = '';
                data.forEach(item => {{
                    if (item.refinement[model] && item.refinement[model].refinement) {{
                        const div = document.createElement('div');
                        div.className = 'comment-pair';
                        div.innerHTML = `
                            <div class="original">${{item.original_comment}}</div>
                            <div class="refined">${{item.refinement[model].refinement}}</div>
                        `;
                        commentsDiv.appendChild(div);
                    }}
                }});
            }}
            updateComments();
            """)
        )

    return HTMLResponse(page.render())
