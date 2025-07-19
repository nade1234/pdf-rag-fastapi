# DWEXO Chatbot

This repository contains the DWEXO Chatbot, a FastAPI application leveraging LangChain and a vector store (Chroma) for Retrieval-Augmented Generation (RAG) to answer questions about the DWEXO enterprise management platform.

## Repository Structure

```
├── .gitlab-ci.yml      # GitLab CI/CD pipeline configuration
├── .gitignore
├── app/
│   ├── Dockerfile      # Docker build instructions
│   ├── main.py         # FastAPI app entrypoint
│   ├── embed.py        # Script to embed documents into Chroma
│   ├── query.py        # API handlers for querying the vector DB
│   ├── utils.py        # Shared utilities (DB, settings)
│   └── requirements.txt
├── data/               # Source documents for ingestion (ignored via .gitignore)
├── venv/               # Local Python virtual environment (ignored)
└── README.md           # This file
```

## Prerequisites

* Python 3.9+
* Docker & Docker Compose
* GitLab account with Container Registry access

## Local Setup

1. **Clone the repo**

   ```bash
   git clone https://gitlab.com/cherry-soft/dwexo-chatbot.git
   cd dwexo-chatbot
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   pip install --upgrade pip
   pip install -r app/requirements.txt
   ```

3. **Start the FastAPI server**

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to explore the OpenAPI UI.

## Embedding Documents

Publish or update your vector store by running:

```bash
python app/embed.py
```

## Docker Usage

1. **Build the Docker image**

   ```bash
   docker build -t registry.gitlab.com/cherry-soft/dwexo-chatbot/my-app:latest -f app/Dockerfile .
   ```

2. **Run the container locally**

   ```bash
   docker run -p 8000:8000 registry.gitlab.com/cherry-soft/dwexo-chatbot/my-app:latest
   ```

3. **Push to GitLab Container Registry**

   ```bash
   docker login registry.gitlab.com
   docker push registry.gitlab.com/cherry-soft/dwexo-chatbot/my-app:latest
   ```

## Continuous Integration / Deployment

The included `.gitlab-ci.yml` will automatically:

* Build and test the Docker image
* Push `:latest` to the Container Registry on merges to `main`

## Contributing

1. Create a new feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
2. Make your changes, commit, and push:

   ```bash
   git commit -am "Add new feature"
   git push -u origin feature/your-feature
   ```
3. Open a Merge Request on GitLab.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
