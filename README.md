# 📄 PDF Chatbot with Gemini API

An intelligent PDF Question-Answering application built using **Streamlit**, **FAISS**, **Sentence Transformers**, and **Google Gemini API**.

This application allows users to upload a PDF and ask questions based on its content using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

- Upload any PDF document  
- Automatic text extraction and chunking  
- Vector embeddings using `all-MiniLM-L6-v2`  
- FAISS vector database for similarity search  
- Context-aware answers using Gemini API  
- Conversational memory support  

---

## 🛠 Tech Stack

- **Streamlit**
- **FAISS**
- **Sentence Transformers**
- **Google Generative AI (Gemini 2.5 Flash)**
- **PyPDF**
- **NumPy**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Manuboby02/PDF-Chat-Bot.git
cd PDF-Chat-Bot
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit faiss-cpu sentence-transformers pypdf google-generativeai numpy
```

### 4️⃣ Set Environment Variable

```bash
setx GOOGLE_API_KEY "your_api_key_here"
```

Restart your terminal after setting the API key.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔐 Security Note

The Google API key is loaded securely using environment variables and is not stored inside the repository.

---

## 📌 Future Improvements

- Improved chunking strategy  
- Persistent vector storage  
- Multi-PDF support  
- Cloud deployment (Streamlit Cloud / AWS)  

---

## 👨‍💻 Author

Manu Boby