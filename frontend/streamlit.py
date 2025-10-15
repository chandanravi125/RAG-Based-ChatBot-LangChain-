import streamlit as st
import requests

# Flask API URL
API_URL = "http://localhost:5000/ask"

def main():
    st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="centered")

    st.title("⚡ Gemini RAG Chatbot")
    st.caption("Ask me anything about NEC guidelines or Wattmonk policies!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat container
    chat_container = st.container()

    # Input box at bottom
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("You:", placeholder="Type your question here...")
        send = st.form_submit_button("Send")

    if send and query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                if response.status_code == 200:
                    answer = response.json().get("response", "No response received.")
                    st.session_state.chat_history.append(
                        {"query": query, "response": answer}
                    )
                else:
                    st.error("❌ Error from backend API.")
            except requests.exceptions.ConnectionError:
                st.error("🚫 Unable to connect to Flask API. Make sure it's running (python main.py).")

    # Display chat history
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {chat['query']}")
            st.markdown(f"**🤖 Bot:** {chat['response']}")
            st.markdown("---")

if __name__ == "__main__":
    main()

