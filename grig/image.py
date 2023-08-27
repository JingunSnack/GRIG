import openai


system_prompt = """
You are a DALL-E Image Prompt Designer, specialized in crafting creative yet concise prompts for
generating DALL-E images that summarize complex analytical findings.

Your role is to translate sentiment analysis, topic identification, and notable mentions into an
image prompt within a 400-character limit.
"""


def generate(summary):
    prompt = _create_prompt(summary)
    return _create_image_url(prompt)


def _create_prompt(summary):
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
                    "Based on the following analytical findings, "
                    "please create a concise DALL-E image prompt:"
                ),
            },
        ],
    )
    return res["choices"][0]["message"]["content"]


def _create_image_url(prompt):
    res = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    return res["data"][0]["url"]
