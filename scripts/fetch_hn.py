import requests
import tqdm
import yaml
from pydantic import BaseModel
from typing import Optional, List


class Topic(BaseModel):
    id: Optional[int] = None
    url: str
    title: str
    description: str


class Comment(BaseModel):
    id: Optional[int] = None
    topic_id: int
    user_id: str
    parent_id: int
    comment: str


# Hacker News API URLs
HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"


def fetch_top_stories(limit=20) -> List[Topic]:
    response = requests.get(HN_TOP_STORIES_URL)
    story_ids = response.json()[:limit]
    topics = []

    for story_id in tqdm.tqdm(story_ids):
        story_data = requests.get(HN_ITEM_URL.format(story_id)).json()
        topic = Topic(
            id=story_data.get("id"),
            url=story_data.get(
                "url", f"https://news.ycombinator.com/item?id={story_id}"
            ),
            title=story_data.get("title", ""),
            description=story_data.get("text", ""),
        )
        topics.append(topic)

    return topics


def fetch_comments_for_story(story_id: int) -> List[Comment]:
    story_data = requests.get(HN_ITEM_URL.format(story_id)).json()
    comments = []

    def parse_comments(comment_ids, parent_id):
        for comment_id in tqdm.tqdm(comment_ids):
            comment_data = requests.get(HN_ITEM_URL.format(comment_id)).json()
            comment = Comment(
                id=comment_data.get("id"),
                topic_id=story_id,
                user_id=comment_data.get("by", ""),
                parent_id=parent_id,
                comment=comment_data.get("text", ""),
            )
            comments.append(comment)
            # Recursively fetch child comments if they exist
            if "kids" in comment_data:
                parse_comments(comment_data["kids"], comment_id)

    if "kids" in story_data:
        parse_comments(story_data["kids"], story_id)

    return comments


def save_to_yaml(
    topics: List[Topic],
    comments: List[Comment],
    filename: str,
):
    with open(filename, "w") as file:
        yaml_data = {
            "topics": [topic.model_dump() for topic in topics],
            "comments": [comment.model_dump() for comment in comments],
        }
        yaml.dump(yaml_data, file)


def main():
    print("Fetching stories")
    topics = fetch_top_stories(limit=10)

    all_comments = []
    for topic in tqdm.tqdm(topics):
        comments = fetch_comments_for_story(topic.id)
        all_comments.extend(comments)

    save_to_yaml(topics, all_comments, "data/ycombinator.yaml")


if __name__ == "__main__":
    main()