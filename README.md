
# ⚡ AI Financial Advisor

This project is a Streamlit-based AI financial assistant that enables users to analyze retail transaction data using LLMs, embeddings, and visualizations. It also provides WhatsApp message support for reporting key insights.

## 🔧 Features

- 📁 Upload retail transaction data in CSV or Excel formats.
- 💬 Ask natural language questions about your data using an LLM-powered Retrieval QA system.
- 📊 Visualize key financial metrics (transaction amounts, tax by branch, etc.) using Plotly.
- 📲 Automatically send analysis results to a WhatsApp number using Twilio.

## 🧠 Tech Stack

- **Frontend**: Streamlit (with custom dark theme)
- **LLM & Embeddings**:
  - `google/flan-t5-base` (via HuggingFace)
  - `sentence-transformers/all-mpnet-base-v2` (for vector similarity)
- **Vector DB**: FAISS
- **Visualization**: Plotly Express
- **Messaging**: Twilio WhatsApp API

## 🚀 How It Works

1. **Upload Transaction Data**: A `.csv` or `.xlsx` file with fields like transaction ID, date, amount, type, etc.
2. **Data Processing**: Converts transaction data into natural language text for embedding.
3. **Vector Store Creation**: Embeds the data into a FAISS vector store.
4. **QA System**: A RetrievalQA chain answers user queries using relevant document context.
5. **Visual Summary**: Shows bar, pie, and line charts summarizing the last 30 days of transaction data.
6. **Messaging**: Sends answers to a specified WhatsApp number.

## 📝 Example Questions

- "What is the highest amount transaction?"
- "Which branch has the most tax?"
- "Show me the lowest transaction amount."

## 📁 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/ai-financial-advisor.git
    cd ai-financial-advisor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```


## 📌 Notes

- Make sure your Twilio account is WhatsApp-enabled and the number is registered.
- Model inference is run locally via Hugging Face — ensure enough resources are available.

## 📄 License

This project is licensed under the MIT License.
