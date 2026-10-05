# 📚 BookMind AI

**BookMind AI** is an AI-powered PDF Question & Answer assistant that allows users to upload a PDF and ask questions about its content.

It uses **RAG (Retrieval-Augmented Generation)** to find relevant information from the uploaded document and generate accurate answers using an LLM.

## 🚀 Features

* 📄 Upload any PDF
* 🔍 Semantic search using embeddings
* 🧠 RAG-based question answering
* ⚡ FAISS vector search
* 🤖 Groq LLM integration
* 💬 Interactive chat interface
* 🎨 Light Ice Crystal + Neon Blue UI
* 🔐 Secure API key using Streamlit Secrets
* 📚 Supports page-wise PDF processing

## 🛠️ Technologies Used

* Python
* Streamlit
* PyMuPDF
* LangChain
* Sentence Transformers
* FAISS
* Groq API
* NumPy

## 🧠 How It Works

```text
PDF Upload
    ↓
PyMuPDF
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Search
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
AI Answer
```

## 📂 Project Structure

```text
BookMind-AI/
│
├── app.py
├── style.css
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/BookMind-AI.git
```

Open the project folder:

```bash
cd BookMind-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create the following file:

```text
.streamlit/secrets.toml
```

Add your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

**Never upload `secrets.toml` to GitHub.**

## ▶️ Run Locally

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deployment

BookMind AI can be deployed using **Streamlit Community Cloud**.

Add the following secret in the Streamlit deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## 🎯 Use Case

BookMind AI can be used for:

* 📖 Books
* 📑 Research papers
* 🎓 Study material
* 📋 Reports
* 📚 Educational PDFs
* 📄 Documentation

## 👨‍💻 Author

**Muhammad Saim**

AI Engineering | Machine Learning | RAG | Computer Vision

---

⭐ If you find BookMind AI useful, consider giving the repository a star!
