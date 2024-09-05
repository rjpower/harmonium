import os
import openai
import dotenv
import yaml
from insight import app
from insight.database import Topic, Comment, Refinement


def new_client():
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def process_comments(input_file, output_file):
    with open(input_file, "r") as f:
        data = yaml.safe_load(f)
        topic = Topic(**data["topic"])
        comments = data["comments"]

    client = new_client()
    results = []
    prompt = app.DEFAULT_PROMPT

    for comment in comments:
        refined_comments = {}
        parents = []
        for model in app.MODELS:
            refinement = app.refine_comment(
                client, model, prompt, topic, comment, parents=parents
            )
            parents.append(
                Comment(
                    topic_id=-1, user_id=-1, parent_id=-1, comment=refinement.refinement
                )
            )
            refined_comments[f"{model}"] = refinement.model_dump()

        results.append(
            {
                "original_comment": comment,
                "refinement": refined_comments,
            }
        )

    print("done.")
    with open(output_file, "w") as f:
        yaml.safe_dump(results, f)


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=os.environ.get("SECRETS_PATH"))
    process_comments(input_file="data/bad_news.yaml", output_file="data/good_news.yaml")
