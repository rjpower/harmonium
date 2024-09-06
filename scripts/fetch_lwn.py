import argparse
import requests
from bs4 import BeautifulSoup
import yaml
from typing import List, Dict
from insight.database import Topic, Comment

def fetch_lwn_page(url: str) -> BeautifulSoup:
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def extract_topic(soup: BeautifulSoup, url: str) -> Topic:
    title = soup.find('h1').text.strip()
    description = soup.find('div', class_='ArticleText').text.strip()
    return Topic(url=url, title=title, description=description)


def extract_comments(soup: BeautifulSoup, topic_id: int) -> List[Comment]:

    def extract_comment_details(comment_box):
        comment = {}
        comment_title = comment_box.find("h3", class_="CommentTitle").get_text(
            strip=True
        )
        comment_text = comment_box.find("div", class_="FormattedComment").get_text(
            strip=True
        )

        comment["title"] = comment_title
        comment["text"] = comment_text
        comment["replies"] = []

        children = comment_box.find_all("p", recursive=False)[-1]

        nested_comments = children.find_all("details", recursive=False)
        for nested_comment in nested_comments:
            comment["replies"].append(extract_comment_details(nested_comment))

        return comment

    main = soup.body.find("div", class_="middlecolumn")
    main = main.find_all("p", recursive=False)[0]
    comments_section = main.find_all("details", class_="CommentBox", recursive=False)
    comments = [
        extract_comment_details(comment_box) for comment_box in comments_section[:1]
    ]
    return comments


def save_to_yaml(topic: Topic, comments: List[Comment], filename: str):
    data = {
        "topics": [topic.model_dump()],
        "comments": [comment for comment in comments],
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Fetch LWN article and comments")
    parser.add_argument("url", help="URL of the LWN article")
    parser.add_argument(
        "--output", default="data/comment_raw.yaml", help="Output YAML file"
    )
    args = parser.parse_args()

    soup = fetch_lwn_page(args.url)
    topic = extract_topic(soup, args.url)
    comments = extract_comments(soup, 123123)

    save_to_yaml(topic, comments, args.output)
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
