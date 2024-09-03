from fasthtml.common import *
import fasthtml.common as fh


def Alert(message: str, type: str):
    return fh.Div(
        fh.Span(message),
        cls=f"alert alert-{type} alert-dismissible fade show",
        role="alert",
    )


def Ul(*args, **kw):
    # kw["cls"] = kw.get("cls", " ") + "list-group"
    return fh.Ul(*args, **kw)


def Li(*args, **kw):
    # kw["cls"] = kw.get("cls", " ") + "list-group-item"
    return fh.Li(*args, **kw)


def Page(title: str, *args, **kw):
    nav = Nav(
        Div(
            Ul(
                Li(A("Home", href="/", cls="nav-link"), cls="nav-item"),
                Li(A("Settings", href="/settings", cls="nav-link"), cls="nav-item"),
                cls="navbar-nav me-auto mb-2 mb-lg-0",
            ),
        ),
        cls="navbar navbar-expand-lg bg-body-tertiary navbar-fixed-top",
    )
    return Title(title), Main(nav, H1(title), *args, **kw)
