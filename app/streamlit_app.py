import streamlit as st
import requests

st.title("🤖 DWEXO Assistant with Memory")

# 🧠 Initialize memory (per session)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Your message")

if st.button("Send") and question:
    # Append user message to memory
    st.session_state.chat_history.append({"role": "user", "content": question})

    # 🤖 Custom logic for "What did I ask?"
    if "what did i ask" in question.lower():
        past_questions = [
            item["content"]
            for item in st.session_state.chat_history
            if item["role"] == "user" and item["content"].lower() != question.lower()
        ]
        if past_questions:
            history_summary = "\n".join(f"- {q}" for q in past_questions)
            assistant_reply = f"You previously asked:\n{history_summary}"
        else:
            assistant_reply = "You haven't asked anything before this."
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    else:
        # 🛰️ Call FastAPI backend
        try:
            response = requests.post(
                "http://127.0.0.1:8000/query/",
                data={"question": question, "debug": False}
            )
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# 💬 Display full chat
if st.session_state.chat_history:
    st.markdown("### 📝 Conversation History")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"👤 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **DWEXO Assistant:** {msg['content']}")
