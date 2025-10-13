
import streamlit as st
import requests

def main():
    st.title("Chatbot")
    st.subheader("Ask me anything!")

    # Session state to store chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Text input for user query
    query = st.text_input("You:", "")

    # Button to submit query
    if st.button("Send"):
        if query:
            # Send request to Flask API
            response = requests.post("http://localhost:5000/ask", json={"query": query})

            # Check if response was successful
            if response.status_code == 200:
                response_json = response.json()
                answer = response_json.get("response")

                # Append query and response to chat history
                st.session_state.chat_history.append({"query": query, "response": answer})

                # Clear query input
                query = ""

    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['query']}")
        st.write(f"**Bot:** {chat['response']}")

if __name__ == "__main__":
    main()
