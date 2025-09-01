# 📚 PDF RAG FastAPI

A powerful **Retrieval-Augmented Generation (RAG)** system built with FastAPI that enables intelligent querying of PDF documents using vector embeddings and LLM-powered responses.

## 🚀 Features

- **📄 PDF Document Processing**: Upload and automatically process PDF files
- **🔍 Semantic Search**: Advanced vector similarity search using HuggingFace embeddings
- **🤖 AI-Powered Responses**: Generate contextual answers using Groq's LLaMA models
- **🧠 Chat Memory**: Session-based conversation memory for better user experience
- **🌐 Multi-language Support**: Handles English and Tunisian dialect queries
- **📧 Smart Notifications**: Email alerts for unanswered questions
- **⚡ Fast API**: High-performance REST API with automatic documentation
- **🎨 Streamlit Frontend**: User-friendly web interface

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **Vector Database**: ChromaDB for embeddings storage
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **LLM**: Groq API with LLaMA-3 70B model
- **Document Processing**: LangChain, PyPDF
- **Frontend**: Streamlit
- **Email**: SMTP integration

## 📦 Installation

### Prerequisites

- Python 3.8+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/nade1234/pdf-rag-fastapi.git
cd pdf-rag-fastapi
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the root directory:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### 5. Create Required Directories

```bash
mkdir -p data/books
mkdir -p chroma
```

## 🚀 Quick Start

### 1. Start the FastAPI Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`
Interactive docs at: `http://localhost:8000/docs`

### 2. Upload and Process PDFs

```bash
# Upload a PDF file
curl -X POST "http://localhost:8000/upload_pdf/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"

# Embed the uploaded PDFs
curl -X POST "http://localhost:8000/embed_new_pdfs/"
```

### 3. Query the System

```bash
curl -X POST "http://localhost:8000/query/" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "question=What is DWEXO?"
```

### 4. Launch Streamlit Frontend (Optional)

```bash
streamlit run streamlit_app.py
```

Access the web interface at: `http://localhost:8501`

## 📚 API Endpoints

### Document Management

- `POST /upload_pdf/` - Upload PDF files
- `POST /embed_new_pdfs/` - Process and embed uploaded PDFs
- `GET /list_indexed/` - List all indexed documents

### Querying

- `POST /query/` - Ask questions about your documents
- `GET /health` - Health check endpoint



## 🏗️ Project Structure

```
pdf-rag-fastapi/
├── main.py              # FastAPI application
├── embed.py             # Document processing and embedding
├── query.py             # Query handling and response generation
├── utils.py             # Utility functions and configurations
├── streamlit_app.py     # Streamlit frontend
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── data/
│   └── books/          # Upload your PDFs here
├── chroma/             # ChromaDB storage (auto-created)
└── README.md
```

## 🔧 Configuration

### Key Settings in `utils.py`

```python
CHROMA_PATH = "chroma"           # Vector database path
DATA_PATH = "data/books"         # PDF storage path
EMBED_MODEL = "all-MiniLM-L6-v2" # Embedding model
MIN_SCORE = 0.1                  # Minimum relevance score
```

### Supported Languages

- English
- Tunisian Dialect (Tunisian Arabic)
- Automatic language detection

## 🎯 Use Cases

- **Documentation Q&A**: Query internal company documents
- **Research Assistant**: Get insights from research papers
- **Knowledge Base**: Build intelligent FAQ systems
- **Customer Support**: Automate responses from product manuals
- **Educational Tools**: Create interactive learning experiences

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🌟 Features Roadmap

- [ ] Support for more document formats (Word, Excel, etc.)
- [ ] Advanced filtering and metadata search
- [ ] Multi-tenant support
- [ ] Real-time document updates
- [ ] Integration with cloud storage (S3, Google Drive)
- [ ] Advanced analytics and usage metrics


⭐ Don't forget to star this repository if you find it helpful!