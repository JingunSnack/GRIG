import openai

system_prompt = """
You are a Daily Note Composer, specialized in transforming technical or formal analysis into casual,
grammatically correct daily notes.

Your job is to make the content easily digestible and relatable, while retaining the essence and key
points of the analysis.

Please regard 'speaker', 'they', and 'user' as 'I' who is the author of this daily note.
"""


def generate(summary):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": f"Analysis result:\n{summary}",
            },
            {
                "role": "user",
                "content": (
                    "Please summarize the following key findings into a casual daily note:"
                ),
            },
        ],
    )
    return res["choices"][0]["message"]["content"]
