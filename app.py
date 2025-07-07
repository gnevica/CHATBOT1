import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import openai
import io
import os
import difflib

st.set_page_config(page_title="📊 CSV Chatbot (OpenRouter)", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

# ✅ Use OpenRouter API key and endpoint from Streamlit Cloud secrets
openai.api_key = st.secrets["OPENROUTER"]["api_key"]
openai.base_url = "https://openrouter.ai/api/v1"

# Intent detection
def detect_intent(query):
    q = query.lower()
    if any(x in q for x in ["forecast", "predict", "next year", "future"]):
        return "forecast"
    elif any(x in q for x in ["plot", "graph", "trend", "visualize"]):
        return "plot"
    else:
        return "general"

# Fuzzy column name correction
def correct_column_names(query, columns):
    words = query.split()
    corrected = []
    for word in words:
        match = difflib.get_close_matches(word, columns, n=1, cutoff=0.8)
        if match:
            corrected.append(match[0])
        else:
            corrected.append(word)
    return " ".join(corrected)

# Title and Upload
st.title("🤖 Chat with your CSV (OpenRouter)")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.success("✅ CSV uploaded successfully!")
    st.write(st.session_state.df.head())

df = st.session_state.df

# Chat UI
st.subheader("💬 Ask something about your data")
for msg in st.session_state.chat_history:
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['bot']}")

user_input = st.text_input("Your query")

if user_input and df is not None:
    corrected_query = correct_column_names(user_input, df.columns)
    intent = detect_intent(corrected_query)
    column_info = "\n".join([f"- {col}: {str(df[col].dtype)}" for col in df.columns])

    # Chat history prompt
    history_prompt = ""
    for msg in st.session_state.chat_history:
        history_prompt += f"User: {msg['user']}\nAssistant: {msg['bot']}\n"

    if intent == "forecast":
        task_instruction = "Use Prophet to forecast time series data. Save plot as 'forecast.png'."
    elif intent == "plot":
        task_instruction = "Use matplotlib to create a plot. Save plot as 'plot.png'."
    else:
        task_instruction = "Use pandas for data analysis. Store tabular output in `result`."

    # Compose full prompt
    full_prompt = f"""
You are a helpful assistant that analyzes CSV datasets.
Sample Data:
{df.head(5).to_csv(index=False)}

Column Info:
{column_info}

Conversation so far:
{history_prompt}

Current User Query:
{corrected_query}

Instructions:
- {task_instruction}
- Only return executable Python code.
"""

    # Call OpenRouter GPT
    response = openai.ChatCompletion.create(
        model="openrouter/gpt-4",  # Try other OpenRouter models too if needed
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2
    )

    code = response.choices[0].message.content
    with st.expander("🧠 GPT-Generated Code"):
        st.code(code, language="python")

    try:
        local_env = {"df": df.copy(), "plt": plt, "Prophet": Prophet}
        exec(code, local_env)

        output_response = ""

        if os.path.exists("forecast.png"):
            st.image("forecast.png", caption="📈 Forecast")
            output_response = "Here's your forecast."
        elif os.path.exists("plot.png"):
            st.image("plot.png", caption="📊 Plot")
            output_response = "Here's your plot."

        if "result" in local_env:
            st.write(local_env["result"])
            csv = local_env["result"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download result as CSV", csv, "result.csv", "text/csv")
            if not output_response:
                output_response = "Here's your data result."

        if not output_response:
            output_response = "Task completed."

        st.session_state.chat_history.append({
            "user": user_input,
            "bot": output_response
        })

    except Exception as e:
        st.error(f"⚠️ Error executing code: {e}")
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": f"⚠️ Error: {e}"
        })
