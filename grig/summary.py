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


def generate(content):
    prev = ""
    for idx in range(1 + (len(content) // 5000)):
        res = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Please include this analysis result of the previous part: {prev}"
                        if prev
                        else ""
                    ),
                },
                {"role": "user", "content": content[idx * 5000 : (idx + 1) * 5000]},
            ],
        )
        prev = res["choices"][0]["message"]["content"]

    return res["choices"][0]["message"]["content"]
