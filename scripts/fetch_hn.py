import requests
import yaml
from pydantic import BaseModel
from typing import Optional, List, Dict
import threading
import time
import os


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


class TopicFetcher(threading.Thread):
    def __init__(self, topic: Topic):
        super().__init__()
        self.topic = topic
        self.comments: List[Comment] = []
        self.fetched = 0
        self.done = False

    def run(self):
        self._fetch_comments_for_story(self.topic.id)
        self.done = True

    def _fetch_comments_for_story(self, story_id: int):
        story_data = requests.get(HN_ITEM_URL.format(story_id)).json()

        def parse_comments(comment_ids, parent_id):
            for comment_id in comment_ids:
                comment_data = requests.get(HN_ITEM_URL.format(comment_id)).json()
                if comment_data:
                    comment = Comment(
                        id=comment_data.get("id"),
                        topic_id=story_id,
                        user_id=comment_data.get("by", ""),
                        parent_id=parent_id,
                        comment=comment_data.get("text", ""),
                    )
                    self.comments.append(comment)
                    self.fetched += 1
                    # Recursively fetch child comments if they exist
                    if "kids" in comment_data:
                        parse_comments(comment_data["kids"], comment_id)

        if "kids" in story_data:
            parse_comments(story_data["kids"], story_id)


def fetch_top_stories(limit: int = 20) -> List[Topic]:
    response = requests.get(HN_TOP_STORIES_URL)
    story_ids = response.json()[:limit]
    topics = []

    for story_id in story_ids:
        story_data = requests.get(HN_ITEM_URL.format(story_id)).json()
        if story_data:
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


def update_progress(fetchers: List[TopicFetcher]):
    print("Monitoring...")
    while any(not fetcher.done for fetcher in fetchers):
        status_line = "".join(["+" if fetcher.done else "-" for fetcher in fetchers])
        print(status_line)
        time.sleep(1)
    print("Done.")


def save_to_yaml(topics: List[Topic], comments: List[Comment], filename: str):
    with open(filename, "w") as file:
        yaml_data = {
            "topics": [topic.model_dump() for topic in topics],
            "comments": [comment.model_dump() for comment in comments],
        }
        yaml.dump(yaml_data, file)


def main():
    print("Fetching stories")
    topics = fetch_top_stories(limit=30)

    print("Fetching comments")
    all_comments = []
    fetchers = [TopicFetcher(topic) for topic in topics]
    progress_thread = threading.Thread(target=update_progress, args=(fetchers,))
    progress_thread.start()
    for i in range(0, len(topics), 30):
        batch = fetchers[i : i + 30]
        for fetcher in batch:
            fetcher.start()

        for fetcher in batch:
            fetcher.join()

    progress_thread.join()
    all_comments.extend(
        [comment for fetcher in fetchers for comment in fetcher.comments]
    )
    save_to_yaml(topics, all_comments, "data/ycombinator.yaml")
    print(f"Fetched {len(topics)} topics and {len(all_comments)} comments")


if __name__ == "__main__":
    main()
