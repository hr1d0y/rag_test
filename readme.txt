Simple RAG App - setup and run commands

Ollama is mandatory for this project.
Install Ollama first:
https://ollama.com/download

Recommended Python version:
python3.12 --version

macOS

1. Clone the project
git clone <your-repo-url>
cd simple_rag

2. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Start Ollama in Terminal window/tab 1
ollama serve

5. Download the required Ollama model in Terminal window/tab 2
ollama pull qwen2.5

6. Start the app in Terminal window/tab 2
streamlit run app.py

7. Open the app
http://localhost:8501

Inside the app use:
- LLM Engine: Ollama (Native Service)
- Ollama Model: qwen2.5
- Vector DB: Chroma DB (Local Persist)

Windows

1. Clone the project
git clone <your-repo-url>
cd simple_rag

2. Create and activate a virtual environment in PowerShell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1

3. Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

4. Start Ollama in PowerShell window 1
ollama serve

5. Download the required Ollama model in PowerShell window 2
ollama pull qwen2.5

6. Start the app in PowerShell window 2
streamlit run app.py

7. Open the app
http://localhost:8501

Inside the app use:
- LLM Engine: Ollama (Native Service)
- Ollama Model: qwen2.5
- Vector DB: Chroma DB (Local Persist)

Linux

1. Clone the project
git clone <your-repo-url>
cd simple_rag

2. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Start Ollama in terminal 1
ollama serve

5. Download the required Ollama model in terminal 2
ollama pull qwen2.5

6. Start the app in terminal 2
streamlit run app.py

7. Open the app
http://localhost:8501

Inside the app use:
- LLM Engine: Ollama (Native Service)
- Ollama Model: qwen2.5
- Vector DB: Chroma DB (Local Persist)

Optional extra packages for advanced features

Semantic Chunking:
pip install langchain-experimental

Qdrant support:
pip install langchain-qdrant qdrant-client

Milvus Lite support:
pip install langchain-milvus "pymilvus[milvus_lite]"

Install all advanced extras at once:
pip install langchain-experimental langchain-qdrant qdrant-client langchain-milvus "pymilvus[milvus_lite]"
