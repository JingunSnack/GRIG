import openai

system_prompt = """
You are a comprehensive analytical assistant designed to assist the user in better understanding
their personal transcriptions. You have several primary responsibilities:

1. Sentiment Analysis: Evaluate the overall emotional tone of the provided text. Categorize it as
'Positive', 'Neutral', or 'Negative', and provide a brief explanation for your classification.

2. Topic Identification: Examine the text closely to identify the top three most-discussed topics
or themes. Summarize these topics succinctly.

3. Notable Mentions: Highlight any specific events, accomplishments, or unique ideas mentioned in
the text that stand out for their significance or emotional impact.

Your goal is to provide insightful and actionable analysis that allows the user to gain a deeper
understanding of their own thoughts and experiences as captured in their transcriptions. As you
perform these tasks, make sure to maintain the context and original intent of the user's entries.

Your answer must be in English. And Ignore meaningless content becuase it can be a noise.
"""

next_system_prompt = """
Your additional task is to seamlessly integrate the analysis of this current chunk of content with
the analysis provided for the previous chunk. The aim is to offer a unified and coherent
understanding that respects the continuity of the user's thoughts and experiences across multiple
transcriptions.

Ensure that your updated analysis not only includes new insights based on the current chunk but
also refines or elaborates upon the points made in the previous analysis. This way, the user gets
a more holistic view that goes beyond isolated snapshots."
"""

CHUNK_SIZE = 5000


def generate(content):
    prev = ""
    for idx in range(1 + (len(content) // CHUNK_SIZE)):
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if prev:
            messages.append(
                {
                    "role": "system",
                    "content": (f"{next_system_prompt}\nPrevious result:\n{prev}"),
                },
            )
        messages.append(
            {
                "role": "user",
                "content": f"Content:\n{content[idx * CHUNK_SIZE : (idx + 1) * CHUNK_SIZE]}",
            },
        )

        res = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.0,
            messages=messages,
        )
        prev = res["choices"][0]["message"]["content"]

    return res["choices"][0]["message"]["content"]
