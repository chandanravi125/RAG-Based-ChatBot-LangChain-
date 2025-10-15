from langchain_google_genai import ChatGoogleGenerativeAI
import os

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
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.invoke(prompt)
    label = response.content.strip().upper()

    if "NEC" in label:
        return "nec"
    elif "WATTMONK" in label:
        return "wattmonk"
    else:
        return "general"
