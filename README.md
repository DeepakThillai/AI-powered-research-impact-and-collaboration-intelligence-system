# ResearchIQ

ResearchIQ is an AI-powered research intelligence platform for universities and research organizations. It brings paper management, semantic search, research trend discovery, collaboration analysis, and impact prediction into one role-aware dashboard.

The application uses a FastAPI backend and a responsive single-page frontend. Paper metadata is stored in MongoDB, semantic embeddings are persisted in ChromaDB, local Sentence Transformers power retrieval, and Groq generates grounded answers from retrieved research content.

## Features

- JWT authentication with student, faculty, and research-head roles
- PDF upload, text extraction, chunking, and background processing
- Local semantic embeddings using `all-MiniLM-L6-v2`
- ChromaDB vector search with source and relevance information
- Retrieval-augmented research assistant powered by Groq
- Department and publication-year dashboards
- TF-IDF and KMeans research trend detection
- NetworkX co-authorship and collaboration analysis
- Random Forest research-impact prediction
- Research-head reporting and institution-wide analytics
- Responsive frontend with dashboards, charts, tables, and upload workflows
- Interactive OpenAPI documentation through FastAPI

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Frontend | HTML, CSS, JavaScript, Plotly.js |
| API | FastAPI, Uvicorn, Pydantic |
| Authentication | JWT, OAuth2 password flow, Passlib, bcrypt |
| Database | MongoDB Atlas |
| Vector storage | ChromaDB |
| Embeddings | Sentence Transformers, PyTorch |
| LLM | Groq API |
| Machine learning | scikit-learn, NumPy, SciPy |
| Document processing | PyPDF2, pdfminer.six |
| Graph analytics | NetworkX |

## Architecture

```text
Browser
  |
  v
FastAPI application
  |-- Authentication and role authorization
  |-- Paper upload and PDF processing
  |-- Dashboard and analytics endpoints
  `-- RAG chat and semantic search
        |
        |-- MongoDB: users, metadata, analytics
        |-- ChromaDB: document chunks and vectors
        |-- Sentence Transformers: local embeddings
        `-- Groq: grounded response generation
```

Paper ingestion follows this pipeline:

```text
PDF -> text extraction -> chunking -> local embeddings -> ChromaDB
                       `-> metadata and status -> MongoDB
```

## Prerequisites

- Python 3.10-3.12 recommended for the pinned ML dependencies
- A MongoDB Atlas deployment or compatible MongoDB connection
- A Groq API key for AI chat responses
- Several GB of free disk space for PyTorch, models, dependencies, and local data

The embedding model is downloaded on first use and then runs locally. Core dashboards can start without a valid Groq key, but `/api/chat/ask` requires one.

## Local Setup

1. Clone the repository and enter the project directory.

   ```bash
   git clone <repository-url>
   cd AI-powered-research-impact-and-collaboration-intelligence-system
   ```

2. Create and activate a virtual environment.

   Windows PowerShell:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create the environment file.

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   macOS or Linux:

   ```bash
   cp .env.example .env
   ```

5. Update `.env` with your MongoDB connection, Groq key, and a strong JWT secret.

   ```env
   GROQ_API_KEY=gsk_your_key
   LLM_MODEL=openai/gpt-oss-120b

   MONGODB_URL=mongodb+srv://USERNAME:PASSWORD@HOST/
   MONGODB_DB_NAME=research_impact_db

   JWT_SECRET_KEY=replace_with_a_long_random_secret
   JWT_ALGORITHM=HS256
   JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

   PAPERS_STORAGE_PATH=./data/papers
   CHROMA_PERSIST_PATH=./data/chromadb
   EMBEDDING_MODEL=all-MiniLM-L6-v2

   APP_HOST=0.0.0.0
   APP_PORT=8000
   DEBUG=True
   ```

   Generate a JWT secret with:

   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

6. Initialize MongoDB and create the demo users.

   ```bash
   python scripts/init_db.py
   ```

7. Optionally generate and index 50 sample research papers.

   ```bash
   python scripts/generate_sample_papers.py
   ```

8. Start the application.

   ```bash
   python main.py
   ```

Open the application at [http://localhost:8000](http://localhost:8000).

## Demo Accounts

The database initialization script creates the following local demonstration accounts:

| Role | Email | Password |
| --- | --- | --- |
| Research Head | `head@university.edu` | `password123` |
| Faculty | `alice@university.edu` | `password123` |
| Faculty | `bob@university.edu` | `password123` |
| Student | `charlie@university.edu` | `password123` |
| Student | `diana@university.edu` | `password123` |

These credentials are for development only. Remove or replace them before deploying the application.

## Application URLs

| URL | Purpose |
| --- | --- |
| `http://localhost:8000/` | ResearchIQ frontend |
| `http://localhost:8000/docs` | Swagger API documentation |
| `http://localhost:8000/redoc` | ReDoc API documentation |
| `http://localhost:8000/health` | Application and MongoDB health check |

## Main API Routes

| Prefix | Purpose |
| --- | --- |
| `/api/auth` | Registration, login, and current-user profile |
| `/api/papers` | PDF upload, paper listing, details, and processing status |
| `/api/chat` | RAG questions, suggestions, and semantic search |
| `/api/dashboard` | Overview, trends, collaboration, impact, and leadership reports |

Protected endpoints expect a bearer token:

```http
Authorization: Bearer <access-token>
```

## Project Structure

```text
.
|-- backend/
|   |-- auth/               # Authentication and authorization
|   |-- dashboard/          # Dashboard and analytics routes
|   |-- ml/                 # Impact, trend, and network analysis
|   |-- papers/             # Upload and PDF processing pipeline
|   |-- rag/                # Retrieval-augmented generation
|   |-- vectordb/           # Embeddings and ChromaDB management
|   |-- config.py           # Environment configuration
|   `-- database.py         # MongoDB connection management
|-- data/
|   |-- chromadb/           # Local vector database persistence
|   |-- logs/               # Rotating application logs
|   `-- papers/             # Uploaded PDF storage
|-- frontend/
|   `-- index.html          # Responsive single-page interface
|-- scripts/
|   |-- init_db.py          # Database indexes and demo users
|   `-- generate_sample_papers.py
|-- .env.example
|-- main.py                 # FastAPI entry point
`-- requirements.txt
```

## Data Reset

To drop the application collections, clear ChromaDB, and recreate the demo users:

```bash
python scripts/init_db.py --reset
```

This is destructive and should only be used against a development database.

## Troubleshooting

**MongoDB shows as disconnected**

Confirm `MONGODB_URL` is valid, the database user has access, and your current IP address is allowed by MongoDB Atlas Network Access.

**The first embedding operation is slow**

Sentence Transformers downloads the configured model on first use. Later requests use the local model cache.

**AI chat reports missing configuration**

Set a valid `GROQ_API_KEY` in `.env` and restart the server. A placeholder key is intentionally treated as unconfigured.

**PowerShell blocks virtual-environment activation**

For the current terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
