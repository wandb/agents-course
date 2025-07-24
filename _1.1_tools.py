import weave
from openai import OpenAI
from pydantic import BaseModel

import config

weave.init(config.WEAVE_PROJECT)
client = OpenAI()


class SentenceEntities(BaseModel):
    entities: list[str]


@weave.op()
def response(instructions: str, user_input: str):
    response = client.responses.parse(
        model="gpt-4.1",
        instructions=instructions,
        input=user_input,
        text_format=SentenceEntities,
    )
    return response.output_parsed


@weave.op()
def process_transcript(transcript: str):
    summary = response("Summarize into 3-5 sentences", transcript)
    tone = response("Determine the tone of the transcript", transcript)
    return summary, tone


@weave.op()
def chapter_1_point_1_tools():
    transcript = "Hello, how are you? I am visiting from Tokyo, and this is my first time here. Do you have any recommendations for places to visit? I am particularly interested in historic sites and aspects of local culture. I'd love to hear your advice for my trip!"
    summary, tone = process_transcript(transcript)
    print(f"summary: {summary}")
    print(f"tone: {tone}")


if __name__ == "__main__":
    chapter_1_point_1_tools()
