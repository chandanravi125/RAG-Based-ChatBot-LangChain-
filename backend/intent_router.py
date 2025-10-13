from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

INTENT_PROMPT = """
You are an AI intent classifier for a hybrid chatbot.

Decide if a user query is about:
1. NEC electrical code or standards  → label: NEC
2. Wattmonk company, its policies, services, pricing, internal info → label: WATTMONK
3. Anything else (casual chat or unrelated) → label: GENERAL

Return only one label: NEC, WATTMONK, or GENERAL.

Examples:
User: What is Article 310 in NEC?
Label: NEC

User: Tell me about Wattmonk's solar installation services.
Label: WATTMONK

User: Who won the cricket match yesterday?
Label: GENERAL

Now classify:
User: {query}
Label:

"""
def detect_intent(query: str) -> str:
    prompt = INTENT_PROMPT.format(query=query)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    label = response.choices[0].message.content.strip().upper()
    if "NEC" in label:
        return "nec"
    elif "WATTMONK" in label:
        return "wattmonk"
    else:
        return "general"
