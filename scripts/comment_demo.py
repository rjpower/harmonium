import os
import openai
import dotenv
import tqdm
import yaml
from insight import app
from insight.database import Topic, Comment, Refinement


def process_comments(input_file, output_file):
    with open(input_file, "r") as f:
        data = yaml.safe_load(f)
        topic = Topic(**data["topics"][0])
        comments = data["comments"]

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    results = []
    prompt = app.DEFAULT_PROMPT

    def _count(comment):
        return 1 + sum(_count(r) for r in comment.get("replies", []))

    refine_count = sum(_count(c) for c in comments) * len(app.MODELS)
    status = tqdm.tqdm(total=refine_count)

    def process_comment(comment, parents):
        refined_comments = {}
        for model in list(app.MODELS):
            refinement = app.refine_comment(
                client, model, prompt, topic, comment["text"], parents=parents
            )
            refined_comments[f"{model}"] = refinement.model_dump()
            status.update()

        result = {
            "original_comment": comment["text"],
            "refinement": refined_comments,
        }
        results.append(result)

        parent_comment = Comment(
            topic_id=-1, user_id=-1, parent_id=-1, comment=comment["text"]
        )
        for reply in comment.get("replies", []):
            process_comment(reply, parents=parents + [parent_comment])

    for comment in comments:
        process_comment(comment, [])

    with open(output_file, "w") as f:
        yaml.safe_dump(results, f)


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=os.environ.get("SECRETS_PATH"))
    process_comments(input_file="data/lwn.yaml", output_file="data/comment_demo.yaml")
