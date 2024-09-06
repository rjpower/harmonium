import pytest
import logging
from fastapi.testclient import TestClient
from insight.app import app


@pytest.fixture(scope="module")
def client():
    logging.getLogger("fastapi").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn").setLevel(logging.DEBUG)

    with TestClient(app) as client:
        yield client


def test_index(client):
    response = client.get("/")
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}. Response text: {response.text}"

def test_settings(client):
    response = client.get("/settings")
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}. Response text: {response.text}"

def test_new_topic(client):
    response = client.post(
        "/topic/new",
        data={
            "topic_url": "https://example.com",
            "topic_description": "test-0",
            "title": "test-0",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303, (response.headers, response.text)
    topic_id = response.headers["Location"].split("/")[-1]
    return int(topic_id)

def test_topic(client):
    topic_id = test_new_topic(client)
    response = client.get(f"/topic/{topic_id}")
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}. Response text: {response.text}"

def test_comment_box(client):
    topic_id = test_new_topic(client)
    response = client.get(f"/topic/comment_box?topic_id={topic_id}&parent_id={topic_id}")
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}. Response text: {response.text}"


def test_demo(client):
    assert client.get("/demo").status_code == 200


def test_settings_save(client):
    response = client.post(
        "/settings/save",
        data={
            "new_model": "anthropic/claude-3-sonnet",
            "prompt_type": "socrates",
            "new_prompt": "Test prompt"
        }
    )
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}. Response text: {response.text}"

def test_reset_prompt(client):
    response = client.get("/settings/reset_prompt")
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}. Response text: {response.text}"

def test_get_prompt(client):
    response = client.get("/settings/get_prompt?prompt_type=socrates")
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}. Response text: {response.text}"
    assert "textarea" in response.text, f"Expected content not found. Response text: {response.text}"
