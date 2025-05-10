import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import plotly.express as px
from twilio.rest import Client

# Messaging config
MESSAGING_CONFIG = {
    "whatsapp": {
        "twilio_sid": "ACdce8ba069ee9b01d3956070d266dba5e",
        "twilio_token": "7adc416993f674d437001d2ad42dfc4f",
        "twilio_number": "whatsapp:+14155238886"
    }
}


def send_whatsapp_message(body, to):
    if not to.startswith("whatsapp:"):
        to = "whatsapp:" + to
    client = Client(MESSAGING_CONFIG["whatsapp"]["twilio_sid"], MESSAGING_CONFIG["whatsapp"]["twilio_token"])
    message = client.messages.create(body=body, from_=MESSAGING_CONFIG["whatsapp"]["twilio_number"], to=to)
    return message.sid


@st.cache_resource

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200, device=-1)
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df


def prepare_data(df):
    documents = (
            "Transaction ID: " + df['transaction_id'] + "\n" +
            "Mall: " + df['mall_name'] + "\n" +
            "Branch: " + df['branch_name'] + "\n" +
            "Date: " + df['transaction_date'].astype(str) + "\n" +
            "Amount: " + df['transaction_amount'].astype(str) + "\n" +
            "Tax: " + df['tax_amount'].astype(str) + "\n" +
            "Type: " + df['transaction_type'] + "\n" +
            "Status: " + df['transaction_status']
    ).tolist()
    return documents


@st.cache_resource
def create_vector_store(_embeddings, documents):
    return FAISS.from_texts(documents, _embeddings, distance_strategy="COSINE")


def initialize_qa_chain(_vector_store, _llm):
    prompt_template = """Answer concisely based on this data:
    {context}

    Question: {question}
    Short Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT, "verbose": False},
        return_source_documents=False
    )


def get_answer(_qa_chain, question):
    return _qa_chain({"query": question})["result"]


def answer_question(question, df, _qa_chain):
    question = question.lower().strip()

    if "highest" in question and "amount" in question:
        row = df.nlargest(1, 'transaction_amount').iloc[0]
        return f"Highest: {row['transaction_amount']:.2f} JOD at {row['mall_name']} (ID: {row['transaction_id']})"
    elif "lowest" in question and "amount" in question:
        row = df.nsmallest(1, 'transaction_amount').iloc[0]
        return f"Lowest: {row['transaction_amount']:.2f} JOD at {row['mall_name']} (ID: {row['transaction_id']})"
    elif "tax" in question and "branch" in question:
        tax_by_branch = df.groupby('branch_name')['tax_amount'].sum().reset_index()
        row = tax_by_branch.nlargest(1, 'tax_amount').iloc[0]
        return f"Highest tax: {row['tax_amount']:.2f} JOD at {row['branch_name']}"

    return get_answer(_qa_chain, question)


def plot_summary(df):
    st.subheader("📊 Transaction Summary")
    recent_df = df[df['transaction_date'] >= (datetime.now() - timedelta(days=30))]

    tax_by_branch = recent_df.groupby("branch_name")["tax_amount"].sum().reset_index()
    fig1 = px.bar(tax_by_branch, x="branch_name", y="tax_amount", title="Total Tax by Branch (Last 30 Days)",
                  text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)

    amount_by_type = df.groupby("transaction_type")["transaction_amount"].sum().reset_index()
    fig2 = px.pie(amount_by_type, names="transaction_type", values="transaction_amount",
                  title="Amount by Transaction Type")
    st.plotly_chart(fig2, use_container_width=True)

    time_series = df.groupby(df['transaction_date'].dt.date)["transaction_amount"].sum().reset_index()
    fig3 = px.line(time_series, x="transaction_date", y="transaction_amount", title="Transaction Amount Over Time")
    st.plotly_chart(fig3, use_container_width=True)


def main():
    st.set_page_config(page_title="Fast Financial Advisor", layout="wide")

    # 🌙 Dark Theme CSS
    st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
        color: #f1f1f1;
    }
    h1, h2, h3, h4 {
        color: #00ffcc !important;
    }
    .css-1d391kg, .css-1cpxqw2 {
        background-color: #1e1e1e !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 style='text-align: center;'>⚡ AI Financial Advisor</h1>", unsafe_allow_html=True)
    st.markdown("Welcome to your smart assistant for analyzing retail transactions. Upload a file to get started!",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload your transaction file (CSV or Excel)", type=["csv", "xlsx"])
    user_phone_number = st.text_input("Enter your WhatsApp number (+962xxxxxxxxx)")

    if uploaded_file:
        with st.spinner("🔄 Processing data..."):
            df = load_data(uploaded_file)
            embeddings = load_embeddings()
            llm = load_llm()
            documents = prepare_data(df)
            vector_store = create_vector_store(embeddings, documents)
            qa_chain = initialize_qa_chain(vector_store, llm)

        st.success("✅ Data Loaded Successfully!")

        tab1, tab2 = st.tabs(["📊 Visual Analysis", "💬 Ask a Question"])

        with tab1:
            st.subheader("Summary & Insights")
            plot_summary(df)

        with tab2:
            st.subheader("Ask About Your Data")
            st.markdown("Examples: `highest amount`, `compare tax by branch`, `lowest amount transaction`, etc.")
            question = st.text_input("💡 Type your question below:")

            if question:
                with st.spinner("💭 Thinking..."):
                    answer = answer_question(question, df, qa_chain)

                    st.markdown(f"""
                    <div style="background-color: #2c2f33; color: #ffffff; padding: 15px; border-radius: 10px; max-width: 80%; margin-bottom: 10px; border: 1px solid #444;">
                        <strong>🤖 AI Assistant:</strong><br>
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)

                    if user_phone_number:
                        send_whatsapp_message(answer, user_phone_number)
                    else:
                        st.error("❌ Please enter a valid WhatsApp number.")
    else:
        st.info("👈 Please upload a CSV or Excel file to begin.")


if __name__ == "__main__":
    main()
