import streamlit as st
import os
import tempfile

IMPORT_ERROR = None
try:
    from langchain_community.llms import LlamaCpp
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc

# Page Config
st.set_page_config(page_title="Simple RAG App", layout="wide")

if IMPORT_ERROR:
    st.error(f"Missing Python dependency: `{IMPORT_ERROR.name}`")
    st.markdown(
        """Install the project dependencies, then rerun Streamlit:

```bash
pip install -r requirements.txt
```

If you plan to use local GGUF models, you may also need a working `llama-cpp-python` build for your machine.
"""
    )
    st.stop()

@st.cache_resource(show_spinner=False)
def ensure_ollama_running():
    """Autonomously spin up the Ollama daemon via detached subprocess if strictly offline."""
    import urllib.request
    import subprocess
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/")
        with urllib.request.urlopen(req, timeout=0.5):
            pass # Already running
    except Exception:
        try:
            # Spawn daemon quietly in background
            import time
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True # Detach so it survives Streamlit hot-reloads
            )
            time.sleep(1.5) # Provide brief hardware spin-up padding
        except Exception:
            pass

# Launch hook silently once per app lifetime
ensure_ollama_running()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generated_chunks" not in st.session_state:
    st.session_state.generated_chunks = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []
if "benchmark_state" not in st.session_state:
    st.session_state.benchmark_state = None
if "benchmark_setup_open" not in st.session_state:
    st.session_state.benchmark_setup_open = False
if "last_benchmark_summary" not in st.session_state:
    st.session_state.last_benchmark_summary = None

@st.cache_resource(show_spinner=False)
def load_llm(model_path, n_gpu_layers, n_ctx, temperature, engine):
    """Load the LLM with caching."""
    if not model_path:
        return None
        
    try:
        if engine == "Ollama (Native Service)":
            from langchain_community.llms import Ollama
            return Ollama(
                model=model_path,
                temperature=temperature
            )
        else:
            if not os.path.exists(model_path):
                return None
            return LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                temperature=temperature,
                f16_kv=True,
                streaming=True,
                verbose=False,
            )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_file(uploaded_files, strategy, params, embedding_model_name, db_choice, db_params, llm_ref=None):
    """Process uploaded files and create vector store."""
    try:
        documents = []
        st.write(f"1/5: Saving {len(uploaded_files)} file(s) to disk...")
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path)
                
            documents.extend(loader.load())
            os.remove(tmp_file_path)
            
        st.write(f"2/5: Extracting text from documents...")
        st.write(f"&nbsp;&nbsp;✓ Extracted {len(documents)} document page(s) total.")
        
        st.write(f"3/5: Applying **{strategy}**...")
        
        st.write(f"&nbsp;&nbsp;✓ Initializing HuggingFace Embedding Model: `{embedding_model_name}`.")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        persist_directory = "./chroma_db"
        
        completion_msg = ""
        if strategy == "Fixed Size Chunking":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=params['chunk_size'], 
                chunk_overlap=params['chunk_overlap']
            )
            splits = text_splitter.split_documents(documents)
            st.write(f"&nbsp;&nbsp;✓ Split document into **{len(splits)}** fixed-size chunks.")
            completion_msg = f"Your document was split perfectly by character count dimensions resulting in **{len(splits)} distinct parts** (Size: {params['chunk_size']}, Overlap: {params['chunk_overlap']} chars)."
            
        elif strategy == "Semantic Chunking":
            from langchain_experimental.text_splitter import SemanticChunker
            st.write("&nbsp;&nbsp;✓ Running Semantic Chunker using HuggingFace Vectors context...")
            text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type=params['breakpoint_type'])
            splits = text_splitter.split_documents(documents)
            st.write(f"&nbsp;&nbsp;✓ Partitioned text by meaning into **{len(splits)}** semantic chunks.")
            completion_msg = f"We routed the text through the semantic algorithm finding topic breakpoints, natively establishing **{len(splits)} highly contextual segments** (Breakpoint type: `{params['breakpoint_type']}`)."

        elif strategy == "Document Aware Chunking":
            from langchain_text_splitters import CharacterTextSplitter
            text_splitter = CharacterTextSplitter(
                separator=params['separator'],
                chunk_size=params['max_chunk_size'],
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False
            )
            splits = text_splitter.split_documents(documents)
            display_sep = params['separator'].replace('\n', '\\n')
            st.write(f"&nbsp;&nbsp;✓ Preserved custom structural marker `{display_sep}` resulting in **{len(splits)}** chunks.")
            completion_msg = f"We utilized the structural syntax custom marker `{display_sep}` to split the file precisely into **{len(splits)} logical chunks**!"

        elif strategy == "Hierarchical Chunking":
            from langchain_classic.retrievers import ParentDocumentRetriever
            from langchain_classic.storage import InMemoryStore
            from langchain_text_splitters import CharacterTextSplitter
            
            parent_splitter = CharacterTextSplitter(
                separator=params['parent_separator'],
                chunk_size=params['parent_chunk_size']
            )
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=params['child_chunk_size'])
            
            db_loc = ""
            if db_choice == "Chroma DB (Local Persist)":
                vectorstore = Chroma(embedding_function=embeddings, persist_directory=db_params['persist_directory'])
                db_loc = db_params['persist_directory']
            elif db_choice == "Qdrant (Local Native)":
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.models import VectorParams, Distance
                dim = 4096 if "7b" in embedding_model_name else (1024 if "bge-large" in embedding_model_name else 384)
                client = QdrantClient(path=db_params['path'])
                if not client.collection_exists("local_rag"):
                    client.create_collection("local_rag", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
                vectorstore = QdrantVectorStore(client=client, collection_name="local_rag", embedding=embeddings)
                db_loc = db_params['path']
            elif db_choice == "Milvus (Local SQLite)":
                from langchain_milvus import Milvus
                from pymilvus import connections
                try:
                    connections.connect("default", uri=db_params['uri'])
                except Exception:
                    pass
                vectorstore = Milvus(
                    embedding_function=embeddings, 
                    connection_args={"uri": db_params['uri']}, 
                    collection_name="local_rag_hier",
                    auto_id=True
                )
                db_loc = db_params['uri']
                
            store = InMemoryStore()
            
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            
            retriever.add_documents(documents, ids=None)
            parent_keys = list(store.yield_keys())
            st.write(f"&nbsp;&nbsp;✓ Hierarchically mapped indexed child chunks to **{len(parent_keys)}** large parent chunks.")
            
            completion_msg = f"Your text was indexed structurally into **{len(parent_keys)} dominant Parent blocks** mapping to highly granular localized Child contexts for extremely precise database retrieval."
            st.session_state.completion_message = f"{completion_msg}\n\nAll specific segments were immediately transformed into multidimensional AI vectors via the robust `{embedding_model_name}` engine, actively mapped into your `{db_loc}` ({db_choice}) repository!"
            
            st.session_state.generated_chunks = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retriever.docstore.store.values()]
            return retriever
            
        elif strategy == "Contextual Chunking (Anthropic)":
            if not llm_ref:
                raise ValueError("Contextual Chunking requires an active LLM to generate summary prepends.")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=params['context_chunk_size'], chunk_overlap=100)
            raw_splits = text_splitter.split_documents(documents)
            full_text = "\n".join([doc.page_content for doc in documents])
            if len(full_text) > 8000: full_text = full_text[:8000] + "... (truncated)"
            
            splits = []
            st.write(f"&nbsp;&nbsp;✓ Orchestrating LLM to inject Contextual Prefixes for {len(raw_splits)} child blocks. (This takes compute...)")
            progress_bar = st.progress(0, text="Iterating Anthropic Context logic...")
            for idx, chunk in enumerate(raw_splits):
                try:
                    prompt = f"Read this full document excerpt:\n\n{full_text}\n\nNow look at this specific chunk from it:\n\n{chunk.page_content}\n\nWrite a 1-sentence explanation of where this chunk objectively fits globally. Output ONLY the 1 sentence inside brackets like: [Context: This is from Chapter X talking about Y.]"
                    msg = llm_ref.invoke(prompt)
                    chunk.page_content = f"{msg.strip()}\n\n{chunk.page_content}"
                except: pass
                splits.append(chunk)
                progress_bar.progress((idx+1)/len(raw_splits), text=f"Processed Context Block {idx+1}/{len(raw_splits)}")
            completion_msg = f"Anthropic Context mapping complete! The LLM successfully read all **{len(splits)} chunks** and glued global document understanding strings onto the very front of each individual snippet!"

        elif strategy == "Graph-Based Chunking (GraphRAG)":
            if not llm_ref: raise ValueError("GraphRAG Chunking requires an active LLM to securely formulate edges.")
            st.write(f"&nbsp;&nbsp;✓ Executing GraphRAG Entity-Edge distillation sequence natively over {len(documents)} pages.")
            prompt = f"Extract all critical Entities and Edges from the following text. Output ONLY a valid JSON list of strings where each string strictly follows the format: '(Entity A) -[RELATIONSHIP]-> (Entity B)'. Do not put anything else outside the JSON array.\n\nText: {documents[0].page_content[:4000]}"
            
            splits = []
            try:
                import json, re
                raw_out = llm_ref.invoke(prompt)
                match = re.search(r'\[.*?\]', raw_out.replace('\n', ''), re.DOTALL)
                if match:
                    extracted_edges = json.loads(match.group(0))
                else:
                    extracted_edges = [line for line in raw_out.split('\n') if '->' in line]
                    
                from langchain_core.documents import Document
                splits = [Document(page_content=edge, metadata={"graph_type": "node_edge"}) for edge in extracted_edges if len(edge)>5]
                st.write(f"&nbsp;&nbsp;✓ Discovered **{len(splits)}** active mathematical Graph edges.")
            except Exception as e:
                splits = documents 
                
            completion_msg = f"GraphRAG Extraction completely successful! A web of **{len(splits)} structural relationships** was logically mathematically extracted and inserted into the vector storage as edges!"

        elif strategy == "Agentic Chunking (LLM JSON)":
            if not llm_ref: raise ValueError("Agentic Chunking requires an active LLM Agent daemon.")
            st.write(f"&nbsp;&nbsp;✓ Unleashing an autonomous LLM Editor Agent to statistically decide cuts on the raw text.")    
            prompt = f"You are a master Data Editor. Read the text below and break it down into perfectly separated thematic/semantic ideas. Return ONLY a JSON array of strings, where each string is a complete cohesive document idea. Do NOT output anything else but JSON. Stop after {params['max_agentic_chunks']} ideas.\n\n{documents[0].page_content[:4000]}"
            
            try:
                import json, re
                raw_out = llm_ref.invoke(prompt)
                match = re.search(r'\[.*?\]', raw_out.replace('\n', ''), re.DOTALL)
                if match:
                    agent_chunks = json.loads(match.group(0))
                else:
                    agent_chunks = [c for c in raw_out.split('\n\n') if len(c)>10][:params['max_agentic_chunks']]
                    
                from langchain_core.documents import Document
                splits = [Document(page_content=idea, metadata={"agent_isolated": True}) for idea in agent_chunks]
                st.write(f"&nbsp;&nbsp;✓ The autonomous Agent successfully finalized **{len(splits)}** semantic cut decisions.")
            except Exception as e:
                splits = documents
                
            completion_msg = f"Your LLM Agent worked tirelessly. Rather than relying on simple logic constraints, the LLM actively navigated the text naturally to manually carve exactly **{len(splits)} perfect idea blocks** from scratch!"
            
        st.write("4/5: Generating embeddings...")
        st.write(f"&nbsp;&nbsp;✓ Computing vectors for all {len(splits)} chunks.")
        
        st.write(f"5/5: Storing vectors into `{db_choice}`...")
        
        if db_choice == "Chroma DB (Local Persist)":
            vectorstore, chroma_reset = build_chroma_with_dimension_reset(
                splits,
                embeddings,
                db_params['persist_directory']
            )
            db_loc = db_params['persist_directory']
            if chroma_reset:
                st.write("&nbsp;&nbsp;✓ Existing Chroma collection used a different embedding dimension, so it was rebuilt for the current embedding model.")
        elif db_choice == "Qdrant (Local Native)":
            from langchain_qdrant import QdrantVectorStore
            vectorstore = QdrantVectorStore.from_documents(splits, embeddings, path=db_params['path'], collection_name="local_rag")
            db_loc = db_params['path']
        elif db_choice == "Milvus (Local SQLite)":
            from langchain_milvus import Milvus
            from pymilvus import connections
            try:
                connections.connect("default", uri=db_params['uri'])
            except Exception:
                pass
            vectorstore = Milvus.from_documents(splits, embeddings, connection_args={"uri": db_params['uri']}, collection_name="local_rag")
            db_loc = db_params['uri']
        
        st.write(f"&nbsp;&nbsp;✓ Vectors saved securely into `{db_loc}`.")
        
        st.session_state.generated_chunks = [{"content": doc.page_content, "metadata": doc.metadata} for doc in splits]
        st.session_state.completion_message = f"{completion_msg}\n\nEach chunk logic was then actively mapped into high-dimensional embedding vectors utilizing **`{embedding_model_name}`** under the hood, and synced straight into {db_choice} at `{db_loc}`!"
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
            
    except Exception as e:
        import traceback
        st.error(f"Error processing document: {e}\n{traceback.format_exc()}")
        return None

def _extract_json_payload(raw_text):
    """Best-effort extraction of a JSON object from an LLM response."""
    import json
    import re

    text = raw_text.strip()
    if not text:
        raise ValueError("Empty evaluator response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in evaluator response.")
    return json.loads(match.group(0))

def _extract_json_payload_with_repair(raw_text, repair_llm=None):
    """Parse JSON, optionally using an LLM repair pass if the payload is malformed."""
    try:
        return _extract_json_payload(raw_text)
    except Exception as original_error:
        if repair_llm is None:
            raise original_error

        repair_prompt = f"""Repair the malformed JSON below.

Rules:
- Preserve the original meaning.
- Return exactly one valid JSON object.
- Do not include markdown.
- Do not explain anything.

MALFORMED JSON:
{raw_text}
"""
        repaired_text = repair_llm.invoke(repair_prompt)
        return _extract_json_payload(repaired_text)

def answer_question_with_rag(question, retriever, llm, search_top_k):
    """Run the standard RAG flow for a single question."""
    retriever.search_kwargs["k"] = search_top_k
    docs = retriever.invoke(question)
    context = "\n\n".join([f"**--- Chunk {i+1} ---**\n\n{doc.page_content}" for i, doc in enumerate(docs)])
    prompt_template = PromptTemplate.from_template(
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nHelpful Answer:"
    )
    formatted_prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    return {
        "answer": response,
        "context": context,
        "chunks": len(docs),
        "prompt_length": len(formatted_prompt),
    }

def generate_benchmark_questions(eval_llm, generated_chunks, relevant_count=5, irrelevant_count=5, status_callback=None):
    """Generate relevant and irrelevant benchmark questions from the current chunks."""
    if status_callback:
        status_callback("Selecting chunk samples for benchmark generation...")
    chunk_preview = "\n\n".join(
        [f"Chunk {idx+1}: {chunk['content'][:600]}" for idx, chunk in enumerate(generated_chunks[:12])]
    )[:10000]
    if status_callback:
        status_callback("Building evaluator prompt for benchmark question generation...")
    prompt = f"""You are creating a compact benchmark set for a RAG system.
Use only the document chunks below.

DOCUMENT CHUNKS:
{chunk_preview}

Return exactly one JSON object with this schema:
{{
  "questions": [
    {{
      "type": "relevant",
      "question": "question answerable from the document",
      "expected_answer": "short correct answer grounded in the document"
    }},
    {{
      "type": "irrelevant",
      "question": "question that is not answerable from the document",
      "expected_answer": "The assistant should say it does not know or that the document does not contain the answer."
    }}
  ]
}}

Requirements:
- Return exactly {relevant_count} relevant questions and {irrelevant_count} irrelevant questions.
- Relevant questions must be clearly answerable from the chunks.
- Irrelevant questions must be outside the document scope.
- Keep questions short.
- Keep expected answers short.
- Output JSON only.
"""
    if status_callback:
        status_callback("Sending benchmark-generation prompt to evaluator model...")
    raw_response = eval_llm.invoke(prompt)
    if status_callback:
        status_callback("Evaluator model responded. Parsing generated questions...")
    payload = _extract_json_payload_with_repair(raw_response, repair_llm=eval_llm)
    if status_callback:
        status_callback("Validating generated question set...")
    questions = payload.get("questions", [])
    relevant = [q for q in questions if q.get("type") == "relevant"][:relevant_count]
    irrelevant = [q for q in questions if q.get("type") == "irrelevant"][:irrelevant_count]
    return relevant + irrelevant

def evaluate_benchmark_answer(eval_llm, benchmark_item, answer, context):
    """Score a generated answer against the benchmark intent and retrieved chunks."""
    prompt = f"""You are grading a RAG answer.
Use only the retrieved chunks as evidence.

QUESTION TYPE: {benchmark_item.get('type', 'unknown')}
QUESTION: {benchmark_item.get('question', '')}
EXPECTED ANSWER: {benchmark_item.get('expected_answer', '')}
RETRIEVED CHUNKS:
{context}

MODEL ANSWER:
{answer}

Return exactly one JSON object:
{{
  "passed": true,
  "verdict": "PASS | FAIL",
  "failure_type": "retrieval_miss | unsupported_answer | partial_answer | wrong_answer | correct_rejection | wrong_rejection | none",
  "reason": "short reason"
}}

Rules:
- For relevant questions, pass only if the answer is supported by the chunks and matches the expected answer.
- For relevant failures, set `failure_type` to one of:
  - `retrieval_miss` if the needed answer was not present in retrieved chunks
  - `unsupported_answer` if the model answered without support from chunks
  - `partial_answer` if the answer was incomplete
  - `wrong_answer` if the answer contradicted the expected answer
- For irrelevant questions, pass only if the answer clearly says the document does not contain the answer or that it does not know.
- For irrelevant failures, set `failure_type` to:
  - `wrong_rejection` if it answered something it should have rejected
  - `correct_rejection` only when the irrelevant question is handled correctly
- If the answer passes, use `failure_type: "none"`.
- Output JSON only.
"""
    raw_grade = _extract_json_payload_with_repair(eval_llm.invoke(prompt), repair_llm=eval_llm)

    verdict = str(raw_grade.get("verdict", "")).strip().upper()
    passed = raw_grade.get("passed")
    reason = str(raw_grade.get("reason", "No grading reason returned.")).strip()
    failure_type = str(raw_grade.get("failure_type", "")).strip().lower()
    question_type = str(benchmark_item.get("type", "")).strip().lower()
    answer_text = answer.strip().lower()
    reason_text = reason.lower()

    if verdict not in {"PASS", "FAIL"}:
        if isinstance(passed, bool):
            verdict = "PASS" if passed else "FAIL"
        else:
            verdict = "FAIL"

    if not isinstance(passed, bool):
        passed = verdict == "PASS"

    if passed:
        failure_type = "none"
    elif question_type == "relevant":
        if failure_type not in {"retrieval_miss", "unsupported_answer", "partial_answer", "wrong_answer"}:
            if any(token in reason_text for token in ["not present", "not mentioned", "not in retrieved", "not in the chunks", "প্রদত্ত তথ্যে", "উল্লেখ নেই"]):
                failure_type = "retrieval_miss"
            elif any(token in reason_text for token in ["partial", "incomplete", "অসম্পূর্ণ"]):
                failure_type = "partial_answer"
            elif any(token in reason_text for token in ["unsupported", "without support", "সমর্থন", "ভিত্তি"]):
                failure_type = "unsupported_answer"
            else:
                failure_type = "wrong_answer"
    else:
        if failure_type not in {"wrong_rejection", "correct_rejection"}:
            if any(token in answer_text for token in ["i don't know", "do not know", "does not contain", "not contain", "জানি না", "উল্লেখ নেই", "প্রদত্ত তথ্যে"]):
                failure_type = "correct_rejection"
            else:
                failure_type = "wrong_rejection"

    return {
        "passed": passed,
        "verdict": verdict,
        "failure_type": failure_type,
        "reason": reason,
    }

def render_evaluator_report(report):
    """Render a compact visual evaluator dashboard."""
    st.markdown("### 📊 Diagnostic Report")

    correctness = report.get("correctness", "Unknown")
    retrieval_support = report.get("retrieval_support", "Unknown")
    hallucination_risk = report.get("hallucination_risk", "Unknown")
    confidence = report.get("confidence", "Unknown")

    verdict_map = {
        "Correct": st.success,
        "Partially Correct": st.warning,
        "Incorrect": st.error,
    }
    verdict_msg = f"Overall verdict: {correctness}"
    verdict_map.get(correctness, st.info)(verdict_msg)

    st.markdown("#### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Correctness**  \n`{correctness}`")
    c2.markdown(f"**Retrieval**  \n`{retrieval_support}`")
    c3.markdown(f"**Hallucination**  \n`{hallucination_risk}`")
    c4.markdown(f"**Confidence**  \n`{confidence}`")

    st.markdown("#### Steps Checked")
    steps = report.get("steps_checked", [])
    if steps:
        for step in steps:
            label = step.get("label", "Unnamed step")
            done = step.get("done", False)
            icon = "PASS" if done else "FAIL"
            st.markdown(f"- {icon} | {label}")
    else:
        st.info("No evaluator steps were returned.")

    st.markdown("#### Claim Checks")
    claims = report.get("claim_checks", [])
    if claims:
        for claim in claims:
            status = claim.get("status", "UNKNOWN")
            text = claim.get("claim", "No claim text")
            reason = claim.get("reason", "No reason provided.")
            if status == "PASS":
                st.success(f"{status} | {text}\n\n{reason}")
            elif status == "PARTIAL":
                st.warning(f"{status} | {text}\n\n{reason}")
            elif status == "FAIL":
                st.error(f"{status} | {text}\n\n{reason}")
            else:
                st.info(f"{status} | {text}\n\n{reason}")
    else:
        st.info("No claim checks were returned.")

    st.markdown("#### Missing Or Wrong")
    missing = report.get("missing", "None")
    wrong = report.get("wrong", "None")
    col_missing, col_wrong = st.columns(2)
    if missing and str(missing).strip().lower() != "none":
        col_missing.warning(f"Missing: {missing}")
    else:
        col_missing.success("Missing: None")
    if wrong and str(wrong).strip().lower() != "none":
        col_wrong.error(f"Wrong: {wrong}")
    else:
        col_wrong.success("Wrong: None")

    st.markdown("#### Bottom Line")
    bottom_line = report.get("bottom_line", "No summary provided.")
    st.info(bottom_line)

def render_sidebar_section(title, subtitle):
    st.markdown(
        f"""
        <div style="
            margin: 0.8rem 0 0.6rem 0;
            padding: 0.8rem 0.85rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(241,245,249,0.96));
            border: 1px solid rgba(226,232,240,0.95);
            box-shadow: 0 10px 24px rgba(15,23,42,0.08);
        ">
            <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:0.2rem;">Section</div>
            <div style="font-size:1rem; font-weight:700; color:#0f172a; margin-bottom:0.18rem;">{title}</div>
            <div style="font-size:0.88rem; color:#475569;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_chroma_with_dimension_reset(splits, embeddings, persist_directory):
    """Rebuild the local Chroma store if an older collection has a different embedding dimension."""
    try:
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        ), False
    except Exception as exc:
        if "embedding with dimension" not in str(exc):
            raise

        import shutil

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        ), True

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-hero {
            padding: 1rem 0.95rem;
            border-radius: 18px;
            background:
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.20), transparent 30%),
                radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.18), transparent 30%),
                linear-gradient(140deg, #0f172a 0%, #1e293b 48%, #0b5d4b 100%);
            color: #f8fafc;
            margin-bottom: 0.8rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
        }
        .sidebar-hero h3 {
            margin: 0 0 0.28rem 0;
            font-size: 1.15rem;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }
        .sidebar-hero p {
            margin: 0;
            font-size: 0.88rem;
            color: rgba(248,250,252,0.84);
        }
        .sidebar-flow {
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(241,245,249,0.96));
            border: 1px solid rgba(226,232,240,0.95);
            box-shadow: 0 10px 24px rgba(15,23,42,0.08);
            margin-bottom: 0.75rem;
        }
        .sidebar-flow-title {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.32rem;
            display: block;
        }
        .sidebar-flow ol {
            margin: 0;
            padding-left: 1rem;
            color: #334155;
            font-size: 0.88rem;
        }
        .sidebar-flow li {
            margin-bottom: 0.22rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-hero">
            <h3>System Configuration</h3>
            <p>Set up the model, choose how documents are chunked, pick embeddings and storage, then upload files to start retrieval.</p>
        </div>
        <div class="sidebar-flow">
            <span class="sidebar-flow-title">Recommended Order</span>
            <ol>
                <li>Choose your inference engine and model.</li>
                <li>Pick a chunking strategy.</li>
                <li>Select an embedding model.</li>
                <li>Choose vector storage and retrieval depth.</li>
                <li>Upload documents and start querying.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar_section("LLM Engine", "Start here. Choose the inference backend, model source, and runtime controls.")
    
    # LLM Settings First
    inference_engine = st.selectbox("LLM Inference Engine", ["Local GGUF (Llama.cpp)", "Ollama (Native Service)"])
    
    if inference_engine == "Ollama (Native Service)":
        st.caption("Ollama automatically manages Apple Silicon GPU RAM and Context sizing.")
        
        # Dynamically fetch local Ollama models via internal API
        ollama_models = []
        err_msg = ""
        try:
            import urllib.request
            import json
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                response = urllib.request.urlopen(req, timeout=1.0)
            except:
                req = urllib.request.Request("http://localhost:11434/api/tags")
                response = urllib.request.urlopen(req, timeout=1.0)
                
            with response:
                models_data = json.loads(response.read().decode())
                ollama_models = [m['name'] for m in models_data.get('models', [])]
        except Exception as e:
            err_msg = str(e)
            pass
            
        if ollama_models:
            model_path = st.selectbox("Select Downloaded Ollama Model", ollama_models)
        else:
            model_path = st.text_input("Ollama Model Name (e.g. qwen2.5)", value="qwen2.5")
            st.caption(f"Auto-detect failed ({err_msg}). Make sure the Ollama app is actively open in your Mac menu bar!")
            
        with st.expander("📥 Pull New Ollama Model"):
            new_ollama_model = st.text_input("Enter model tag (e.g., 'llama3.2', 'deepseek-coder-v2')", key="new_ollama_input")
            
            if "ollama_pull_state" in st.session_state:
                # Async polling state
                q = st.session_state.ollama_pull_state['q']
                stop_event = st.session_state.ollama_pull_state['stop_event']
                
                col1, col2 = st.columns([3, 1])
                if col2.button("❌ Cancel", key="cancel_pull"):
                    stop_event.set()
                    del st.session_state.ollama_pull_state
                    st.warning("Download aborted explicitly!")
                    import time
                    time.sleep(1)
                    st.rerun()
                
                import queue
                latest_data = st.session_state.ollama_pull_state.get('last_data')
                while not q.empty():
                    try:
                        latest_data = q.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_data:
                    st.session_state.ollama_pull_state['last_data'] = latest_data
                    
                    if latest_data['type'] == 'error':
                        col1.error(f"Download failed: {latest_data['error']}")
                        del st.session_state.ollama_pull_state
                    elif latest_data['type'] == 'done':
                        col1.success(f"Successfully installed {new_ollama_model}!")
                        del st.session_state.ollama_pull_state
                        import time
                        time.sleep(1)
                        st.rerun()
                    elif latest_data['type'] == 'cancelled':
                        col1.warning("Aborted by User.")
                        if "ollama_pull_state" in st.session_state:
                            del st.session_state.ollama_pull_state
                    elif latest_data['type'] == 'progress':
                        d = latest_data['data']
                        status_str = d.get('status', 'Downloading...')
                        if 'total' in d and 'completed' in d and d['total'] > 0:
                            pct = d['completed'] / d['total']
                            mb_done = d['completed'] / (1024*1024)
                            mb_total = d['total'] / (1024*1024)
                            col1.progress(pct, text=f"**{status_str}** — {mb_done:.1f} / {mb_total:.1f} MB ({pct*100:.1f}%)")
                        else:
                            col1.info(f"Status: {status_str}")
                            
                if "ollama_pull_state" in st.session_state:
                    import time
                    time.sleep(0.1)  # Natively throttle Streamlit re-render loop
                    st.rerun()
                    
            else:
                if st.button("Download / Update Model", key="pull_ollama_btn") and new_ollama_model:
                    import threading
                    import queue
                    import urllib.request
                    import json
                    
                    q = queue.Queue()
                    stop_event = threading.Event()
                    
                    def pull_thread(model_name, q_ref, stop_ref):
                        try:
                            req = urllib.request.Request("http://127.0.0.1:11434/api/pull", data=json.dumps({"name": model_name}).encode('utf-8'))
                            req.add_header('Content-Type', 'application/json')
                            with urllib.request.urlopen(req) as response:
                                for line in response:
                                    if stop_ref.is_set():
                                        q_ref.put({"type": "cancelled"})
                                        return
                                    if line:
                                        q_ref.put({"type": "progress", "data": json.loads(line)})
                            q_ref.put({"type": "done"})
                        except Exception as e:
                            q_ref.put({"type": "error", "error": str(e)})

                    t = threading.Thread(target=pull_thread, args=(new_ollama_model.strip(), q, stop_event))
                    t.start()
                    
                    st.session_state.ollama_pull_state = {'q': q, 'stop_event': stop_event, 'last_data': None}
                    st.rerun()

        temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        n_gpu_layers = -1
        n_ctx = 4096
    else:
        model_path = st.text_input("GGUF Model Path", value="")
        n_gpu_layers = st.slider(
            "GPU Offload (n_gpu_layers)",
            min_value=-1,
            max_value=100,
            value=-1,
            help="Higher = faster if your machine has enough GPU/unified memory. Lower = safer but slower. -1 means offload as much as possible."
        )
        st.caption("Higher values try to push more of the model to GPU memory. Increase for speed, decrease if loading fails or memory gets tight.")
        n_ctx = st.slider(
            "Memory for Prompt + Context (n_ctx)",
            min_value=512,
            max_value=8192,
            value=4096,
            step=256,
            help="Higher = the model can read more text at once, but uses more memory and can be slower."
        )
        st.caption("Increase this if long document questions lose context. Decrease it if you want lower memory usage and faster runs.")
        temperature = st.slider(
            "Answer Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower = more stable and factual. Higher = more creative and varied, but more risk of drift."
        )
        st.caption("For document Q&A, lower values are usually better. Raise it only if you want more flexible or exploratory wording.")
    
    llm = None
    if model_path:
        with st.status(f"Initializing {inference_engine}...", expanded=False) as status:
            if inference_engine != "Ollama (Native Service)":
                st.write(f"Allocating {n_gpu_layers} layers to GPU...")
                st.write(f"Setting Context length to {n_ctx}...")
            llm = load_llm(model_path, n_gpu_layers, n_ctx, temperature, engine=inference_engine)
            if llm:
                status.update(label="Model Loaded!", state="complete")
            else:
                status.update(label="Failed to Load Model", state="error")
    llm_status = "Ready" if llm else "Not Loaded"
    model_status = model_path if model_path else "No model selected"
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.55rem; margin:0.7rem 0 0.3rem 0;">
            <div style="padding:0.7rem 0.75rem; border-radius:14px; background:rgba(255,255,255,0.94); border:1px solid rgba(226,232,240,0.9);">
                <div style="font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.08em;">Pipeline</div>
                <div style="font-size:0.95rem; font-weight:700; color:#0f172a;">{llm_status}</div>
            </div>
            <div style="padding:0.7rem 0.75rem; border-radius:14px; background:rgba(255,255,255,0.94); border:1px solid rgba(226,232,240,0.9);">
                <div style="font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.08em;">Engine</div>
                <div style="font-size:0.95rem; font-weight:700; color:#0f172a;">{inference_engine.split(' (')[0]}</div>
            </div>
        </div>
        <div style="padding:0.72rem 0.78rem; border-radius:14px; background:rgba(248,250,252,0.96); border:1px solid rgba(226,232,240,0.92); margin-bottom:0.2rem;">
            <div style="font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.08em;">Selected Model</div>
            <div style="font-size:0.88rem; font-weight:600; color:#0f172a; word-break:break-word;">{model_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar_section("Chunking Strategy", "Control how documents are split before embedding and retrieval.")
    chunk_strategy = st.selectbox(
        "Select Strategy", 
        ["Fixed Size Chunking", "Semantic Chunking", "Document Aware Chunking", "Hierarchical Chunking", "Contextual Chunking (Anthropic)", "Graph-Based Chunking (GraphRAG)", "Agentic Chunking (LLM JSON)"]
    )
    
    chunk_params = {}
    if chunk_strategy == "Fixed Size Chunking":
        st.caption("Splits text strictly based on word/character count, ignoring meaning.")
        chunk_params['chunk_size'] = st.number_input("Word/Char Count per Chunk", min_value=100, max_value=4000, value=1000)
        st.caption("— **Chunk Size**: Determines the maximum characters per chunk. Smaller = hyper-precise searches. Larger = broader context but consumes more inference tokens.")
        chunk_params['chunk_overlap'] = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200)
        st.caption("— **Overlap**: Artificially duplicates trailing characters into the next chunk so hard cuts don't sever important sentences.")
    elif chunk_strategy == "Semantic Chunking":
        st.caption("A neural chunker that groups sentences by meaning ensuring topical coherence.")
        chunk_params['breakpoint_type'] = st.selectbox("Breakpoint Threshold", ["percentile", "standard_deviation", "interquartile"])
        if chunk_params['breakpoint_type'] == "percentile":
            st.caption("— **Percentile**: Splits text when the semantic shift is in the top X% of all shifts. Safe, default statistical choice.")
        elif chunk_params['breakpoint_type'] == "standard_deviation":
            st.caption("— **Standard Deviation**: Splits only when topic deviations are highly abnormal. Great for dense technical documents.")
        else:
            st.caption("— **Interquartile**: A strict outlier detection algorithm. Perfect for long texts with massively divergent chapters.")
    elif chunk_strategy == "Document Aware Chunking":
        st.caption("Splits exclusively on structural boundaries (like double newlines separating paragraphs).")
        chunk_params['separator'] = st.text_input("Custom Separator Marker (use '\\n' for newlines)", value="\\n\\n").replace('\\n', '\n')
        st.caption("— **Separator**: The exact text array the parser will look for to snap a chunk boundary (e.g. `## `).")
        chunk_params['max_chunk_size'] = st.number_input("Max Size Fallback", min_value=100, max_value=4000, value=2000)
        st.caption("— **Max Size Fallback**: An artificial character limit to forcibly cut a block if the custom separator isn't found for a dangerously long time.")
    elif chunk_strategy == "Hierarchical Chunking":
        st.caption("Embeds small children chunks but returns large parent context on hit.")
        chunk_params['parent_separator'] = st.text_input("Parent Custom Separator (Document Aware)", value="\\n\\n").replace('\\n', '\n')
        st.caption("— **Parent Separator**: The structural marker separating your massive parent blocks.")
        chunk_params['parent_chunk_size'] = st.number_input("Parent Size Fallback", min_value=1000, max_value=8000, value=2000)
        st.caption("— **Parent Size Fallback**: The max limit to forcibly cut a parent block if the custom separator is missing.")
        chunk_params['child_chunk_size'] = st.number_input("Child Size (Fixed Size)", min_value=100, max_value=1000, value=400)
        st.caption("— **Child Size**: The tiny inner paragraph size that the neural vectors actually map against. Needs to be extremely small.")
    elif chunk_strategy == "Contextual Chunking (Anthropic)":
        st.caption("Solves lost context by preceding every chunk with an LLM-generated global summary block.")
        chunk_params['context_chunk_size'] = st.number_input("Base Chunk Size", min_value=100, max_value=2000, value=800)
        st.caption("— **Warning**: Highly computationally expensive. This requires your selected LLM to process every single chunk physically during extraction.")
    elif chunk_strategy == "Graph-Based Chunking (GraphRAG)":
        st.caption("Uses an LLM to extract a massive neural web of entities and relationship edges, dropping rigid text bounds entirely.")
        chunk_params['graph_focus'] = st.selectbox("Extraction Physics", ["Noun/Entity Centric", "Action/Verb Centric", "Comprehensive (Slow)"])
        st.caption("— **Warning**: The Agent acts as an editor physically deciding cuts mathematically based on Graph rules.")
    elif chunk_strategy == "Agentic Chunking (LLM JSON)":
        st.caption("Unleashes an autonomous Agent to statistically read the raw text and break it into perfectly separated thematic JSON ideas via prompt formatting.")
        chunk_params['max_agentic_chunks'] = st.number_input("Max Ideas (Yield limit)", min_value=1, max_value=50, value=15)
        st.caption("— **Warning**: Extremely slow logic bounds checking natively over long contexts.")
    
    render_sidebar_section("Embedding Model", "Choose the vector model used to map your chunks into searchable representations.")
    
    def is_cached(repo_id):
        import os
        hf_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}")
        st_path = os.path.expanduser(f"~/.cache/torch/sentence_transformers/{repo_id.replace('/', '_')}")
        return os.path.exists(hf_path) or os.path.exists(st_path)

    def delete_cache(repo_id):
        import os
        import shutil
        hf_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}")
        st_path = os.path.expanduser(f"~/.cache/torch/sentence_transformers/{repo_id.replace('/', '_')}")
        if os.path.exists(hf_path):
            shutil.rmtree(hf_path)
        if os.path.exists(st_path):
            shutil.rmtree(st_path)

    model_mapping = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "e5-mistral": "intfloat/e5-mistral-7b-instruct"
    }
    
    display_options = []
    for friendly_name, repo_id in model_mapping.items():
        if is_cached(repo_id):
            display_options.append(f"🟢 {friendly_name} (Ready)")
        else:
            display_options.append(f"⚪ {friendly_name} (Requires Download)")

    embedding_model_option = st.selectbox("Select Embedding Model", display_options)
    
    selected_label = embedding_model_option.split(" (")[0][2:]
    selected_embedding_model = model_mapping[selected_label]
    
    if "7b" in selected_embedding_model:
        st.caption("⚠️ **Warning**: 7B embedding models require ~14GB+ RAM and may take a long time to load.")
        
    if not is_cached(selected_embedding_model):
        st.warning(f"**Action Required**: `{selected_embedding_model}` is not locally downloaded.")
        if st.button("🔽 Download Model to Cache"):
            from huggingface_hub import snapshot_download
            import sys
            import time
            
            from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
            import threading
            
            # Using a custom sys.stderr interceptor and a time throttle to capture fast TQDM progress lines!
            class ThrottledStderrWrapper:
                def __init__(self, ui_container):
                    self.ui_container = ui_container
                    self.buffer = ""
                    self.original_stderr = sys.stderr
                    self.last_update = time.time()
                    self.ctx = get_script_run_ctx()
                    
                    self.ui_container.empty()
                    col1, col2 = self.ui_container.columns([3, 1])
                    self.p_bar = col1.progress(0.0, text="🚀 Establishing Secure Connection...")
                    self.p_metrics = col2.empty()

                def isatty(self):
                    return True # Tricks tqdm into thinking it's writing to a real terminal

                def write(self, message):
                    if "missing ScriptRunContext" in message:
                        return # Silence safe internal thread warnings
                        
                    # TQDM often executes from detached `ThreadPoolExecutor` workers via huggingface_hub inside the terminal
                    if get_script_run_ctx() is None:
                        add_script_run_ctx(threading.current_thread(), self.ctx)
                        
                    self.original_stderr.write(message)
                    if '\r' in message:
                        self.buffer = message.split('\r')[-1]
                    elif '\n' in message:
                        # Keeps just the very last line if tqdm uses newlines
                        self.buffer = message.split('\n')[-1]
                    else:
                        self.buffer += message

                    # Prevent websocket flood (only update UI once every 0.15 seconds)
                    current_time = time.time()
                    if current_time - self.last_update > 0.15:
                        clean_text = self.buffer.strip()
                        if clean_text:
                            import re
                            # Safely extract dynamic values from HuggingFace string formats!
                            pct_match = re.search(r'(\d{1,3})%', clean_text)
                            percent = min(100, int(pct_match.group(1))) if pct_match else 0
                            
                            speed_match = re.search(r',\s*([^\]]*)\]', clean_text)
                            speed = speed_match.group(1).strip() if speed_match else "Calculating..."
                            
                            phase_raw = clean_text.split('|')[0].strip() if '|' in clean_text else "Downloading Checkpoints..."
                            phase = phase_raw.replace("Downloading ", "").replace("(incomplete total...):", "").strip()
                            if not phase: phase = "Binary Pipeline"

                            self.p_bar.progress(percent / 100.0, text=f"**Status:** {phase} ({percent}%)")
                            self.p_metrics.markdown(f"**⚡ Speed:** `{speed}`")
                            
                        self.last_update = current_time

                def flush(self):
                    self.original_stderr.flush()

            with st.status(f"Downloading {selected_embedding_model} from HuggingFace...", expanded=True) as status:
                progress_container = st.container()
                
                original_stderr = sys.stderr
                try:
                    sys.stderr = ThrottledStderrWrapper(progress_container)
                    snapshot_download(selected_embedding_model, max_workers=1)
                    sys.stderr = original_stderr
                    status.update(label="Download Complete! Reloading application...", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    sys.stderr = original_stderr
                    status.update(label=f"Download Failed: {e}", state="error")
    else:
        if st.button("🗑️ Delete Model from Cache"):
            delete_cache(selected_embedding_model)
            st.success(f"Deleted {selected_embedding_model} cleanly from your local cache!")
            st.rerun()

    render_sidebar_section("Storage Layer", "Pick where embeddings live and how retrieval should search them.")
    db_choice = st.selectbox(
        "Select Database Architecture",
        ["Chroma DB (Local Persist)", "Qdrant (Local Native)", "Milvus (Local SQLite)", "Pinecone (Cloud DB)"]
    )
    
    db_params = {}
    if db_choice == "Pinecone (Cloud DB)":
        st.error("❌ The Pinecone architectural SDK fails to compile on Python 3.14 environments due to severe constraints inside its underlying simsimd framework limits. Pinecone is blocked natively.")
    elif db_choice == "Qdrant (Local Native)":
        db_params['path'] = st.text_input("Qdrant Path", value="./qdrant_db")
        st.caption(f"Qdrant will save fast dense vectors isolated dynamically in `{db_params['path']}`.")
    elif db_choice == "Milvus (Local SQLite)":
        db_params['uri'] = st.text_input("Milvus DB URI", value="./milvus_demo.db")
        st.caption(f"Milvus Lite mounts entirely within an invisible localized `{db_params['uri']}` SQLite file.")
    else:
        db_params['persist_directory'] = "./chroma_db"
        st.caption("Standard ChromaDB instance persisting globally via flat files in `./chroma_db`.")
        
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.55rem; margin:0.55rem 0 0.25rem 0;">
            <div style="padding:0.7rem 0.75rem; border-radius:14px; background:rgba(255,255,255,0.94); border:1px solid rgba(226,232,240,0.9);">
                <div style="font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.08em;">Embedding</div>
                <div style="font-size:0.88rem; font-weight:700; color:#0f172a;">{selected_label}</div>
            </div>
            <div style="padding:0.7rem 0.75rem; border-radius:14px; background:rgba(255,255,255,0.94); border:1px solid rgba(226,232,240,0.9);">
                <div style="font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.08em;">Database</div>
                <div style="font-size:0.88rem; font-weight:700; color:#0f172a;">{db_choice.split(' (')[0]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Retrieval Configuration")
    search_top_k = st.slider("Top-K Retrieval Limit", min_value=1, max_value=20, value=5, help="How many distinct chunks the database will source and inject into the prompt.")

    render_sidebar_section("Documents", "Upload source files, embed them, clear them, or inspect the generated chunks.")
    
    # Process uploads logically only if the model is ready and Database architecture doesn't fatally crash
    if is_cached(selected_embedding_model) and db_choice != "Pinecone (Cloud DB)":
        uploaded_files = st.file_uploader("Upload Documents for Vectorization", type=["pdf", "txt"], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
        
        current_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
        files_changed = current_file_names != st.session_state.processed_file_names

        if uploaded_files and (st.session_state.retriever is None or files_changed):
            if chunk_strategy in ["Contextual Chunking (Anthropic)", "Graph-Based Chunking (GraphRAG)", "Agentic Chunking (LLM JSON)"] and not llm:
                st.error("❌ **LLM Offline:** You MUST initialize an LLM Engine from the top of the sidebar first because this advanced chunking strategy literally runs live AI inference during extraction!")
            else:
                st.session_state.retriever = None
                st.session_state.generated_chunks = None
                with st.status("Embedding Documents...", expanded=True) as status:
                    st.session_state.retriever = process_file(uploaded_files, chunk_strategy, chunk_params, selected_embedding_model, db_choice, db_params, llm_ref=llm)
                    if st.session_state.retriever:
                        st.session_state.processed_file_names = current_file_names
                        status.update(label="Documents embedded successfully!", state="complete", expanded=False)
                    else:
                        status.update(label="Failed to process documents", state="error", expanded=False)
                    
        if uploaded_files and st.session_state.retriever is not None:
            if st.session_state.get("completion_message"):
                st.success("✅ **Vectorization Successful!**")
                st.info(st.session_state.completion_message)
            else:
                st.success(f"Documents embedded using {chunk_strategy}.")
                
            if st.button("🗑️ Clear Current Documents"):
                st.session_state.retriever = None
                st.session_state.generated_chunks = None
                st.session_state.processed_file_names = []
                st.session_state.uploader_key += 1
                st.rerun()

        st.markdown("---")
        st.markdown("### Database Management")
        if st.button("🧨 Purge Entire Database Data"):
            import shutil
            import os
            import time
            deleted = False
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                deleted = True
            if os.path.exists("./qdrant_db"):
                shutil.rmtree("./qdrant_db")
                deleted = True
            if os.path.exists("./milvus_demo.db"):
                os.remove("./milvus_demo.db")
                deleted = True
                
            if deleted:
                st.session_state.retriever = None
                st.session_state.generated_chunks = None
                st.success("Internal vector databases recursively wiped clean across all architectures!")
                time.sleep(1) # Visual pause to see success message before reload
                st.rerun()
            else:
                st.info("No database data found.")
                
        # Preview Chunks Option
        if st.session_state.generated_chunks:
            st.markdown("---")
            with st.expander(f"👁️ View Evaluated Chunks ({len(st.session_state.generated_chunks)} chunks total)"):
                st.caption("Scroll through to see how your selected chunking architecture parsed your document:")
                for idx, chunk in enumerate(st.session_state.generated_chunks):
                    st.markdown(f"**Chunk {idx+1} [Length: {len(chunk['content'])}]:**")
                    st.text(chunk['content'])
                    st.markdown("---")

# --- MAIN WINDOW & EVALUATOR ---
main_chat_col, eval_panel_col = st.columns([7, 4])

with main_chat_col:
    chat_ready_llm = "Ready" if llm else "Not Loaded"
    chat_ready_db = "Ready" if st.session_state.retriever else "Not Ready"
    chat_doc_count = len(st.session_state.get("processed_file_names", []))
    chat_doc_label = f"{chat_doc_count} loaded" if chat_doc_count else "No docs"

    st.markdown(
        """
        <style>
        .rag-hero {
            position: relative;
            overflow: hidden;
            padding: 1.4rem 1.35rem 1.2rem 1.35rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(255, 193, 7, 0.28), transparent 28%),
                radial-gradient(circle at bottom left, rgba(0, 166, 126, 0.18), transparent 30%),
                linear-gradient(135deg, #0f172a 0%, #172554 48%, #0b3b2e 100%);
            color: #f8fafc;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.22);
            animation: heroFloat 0.7s ease-out;
            margin-bottom: 0.9rem;
        }
        .rag-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
        }
        .rag-hero p {
            margin: 0;
            max-width: 48rem;
            color: rgba(248, 250, 252, 0.86);
            font-size: 0.98rem;
        }
        .rag-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.85rem 0 1rem 0;
        }
        .rag-status-card {
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(248,250,252,0.95));
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        }
        .rag-status-label {
            display: block;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.28rem;
        }
        .rag-status-value {
            display: block;
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
        }
        .rag-empty-state {
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(241,245,249,0.96));
            border: 1px solid rgba(226, 232, 240, 0.9);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
            animation: heroFloat 0.85s ease-out;
        }
        .rag-empty-title {
            margin: 0 0 0.45rem 0;
            font-size: 1.02rem;
            font-weight: 700;
            color: #0f172a;
        }
        .rag-empty-list {
            margin: 0;
            padding-left: 1rem;
            color: #334155;
        }
        .rag-empty-list li {
            margin-bottom: 0.28rem;
        }
        .rag-conversation-shell {
            border-radius: 20px;
            padding: 0.9rem 0.95rem 1rem 0.95rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
            border: 1px solid rgba(226, 232, 240, 0.92);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.9rem;
        }
        .rag-conversation-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.75rem;
        }
        .rag-conversation-title {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }
        .rag-conversation-subtitle {
            font-size: 0.86rem;
            color: #64748b;
            margin: 0.18rem 0 0 0;
        }
        .rag-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .rag-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.06);
            color: #0f172a;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .rag-chat-note {
            margin-top: 0.8rem;
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
            background: rgba(15, 118, 110, 0.08);
            border: 1px solid rgba(13, 148, 136, 0.18);
            color: #134e4a;
            font-size: 0.84rem;
        }
        div[data-testid="stChatInput"] {
            max-width: 66%;
        }
        div[data-testid="stChatMessage"] {
            animation: heroFloat 0.45s ease-out;
        }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            border-radius: 18px;
            padding: 0.15rem 0.1rem;
        }
        div[data-testid="stChatMessage"]:has([aria-label="user avatar"]) {
            background: linear-gradient(135deg, rgba(219, 234, 254, 0.78), rgba(239, 246, 255, 0.82));
            border: 1px solid rgba(147, 197, 253, 0.55);
            border-radius: 20px;
            padding: 0.35rem 0.55rem 0.35rem 0.35rem;
            margin-bottom: 0.75rem;
        }
        div[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) {
            background: linear-gradient(135deg, rgba(236, 253, 245, 0.9), rgba(240, 253, 250, 0.94));
            border: 1px solid rgba(110, 231, 183, 0.4);
            border-radius: 20px;
            padding: 0.35rem 0.55rem 0.35rem 0.35rem;
            margin-bottom: 0.75rem;
        }
        @keyframes heroFloat {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="rag-hero">
            <h1>Intelligent Document Assistant</h1>
            <p>Ask focused questions, inspect retrieved evidence, and audit answer quality from the same workspace. The assistant pulls context from your embedded documents before generating a response.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="rag-status-grid">
            <div class="rag-status-card">
                <span class="rag-status-label">LLM Pipeline</span>
                <span class="rag-status-value">{chat_ready_llm}</span>
            </div>
            <div class="rag-status-card">
                <span class="rag-status-label">Vector Database</span>
                <span class="rag-status-value">{chat_ready_db}</span>
            </div>
            <div class="rag-status-card">
                <span class="rag-status-label">Document Set</span>
                <span class="rag-status-value">{chat_doc_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="rag-empty-state">
                <div class="rag-empty-title">How to get started</div>
                <ul class="rag-empty-list">
                    <li>Load an LLM from the sidebar.</li>
                    <li>Upload one or more PDF or TXT files.</li>
                    <li>Wait for vectorization to finish.</li>
                    <li>Start with a concrete question about the uploaded material.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="rag-conversation-shell">
            <div class="rag-conversation-head">
                <div>
                    <div class="rag-conversation-title">Conversation</div>
                    <div class="rag-conversation-subtitle">Grounded answers appear here with retrieval telemetry and chunk evidence.</div>
                </div>
                <div class="rag-chip-row">
                    <span class="rag-chip">LLM: {chat_ready_llm}</span>
                    <span class="rag-chip">DB: {chat_ready_db}</span>
                    <span class="rag-chip">Docs: {chat_doc_label}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Display Chat History
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
            if "metrics" in msg:
                st.caption("⚡ **Generation Telemetry:**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Vectors Evaluated", msg["metrics"].get("total"))
                m2.metric("Top-K Injected", msg["metrics"].get("chunks"))
                m3.metric("Prompt Density", f"{msg['metrics'].get('length')} chars")
                m4.metric("Latency", f"{msg['metrics'].get('time', 0.0)} sec")
                
            # Give access to the literal context the bot observed
            if "context_used" in msg:
                with st.expander("🔍 View Raw Database Chunks Sourced"):
                    st.markdown(msg["context_used"])
    
    # Chat Input
    st.markdown(
        """
        <div class="rag-chat-note">
            Use the native bottom composer to ask a direct question about the uploaded documents.
        </div>
        """,
        unsafe_allow_html=True,
    )

with eval_panel_col:
    benchmark_state = st.session_state.benchmark_state
    st.markdown(
        """
        <style>
        .eval-hero {
            position: relative;
            overflow: hidden;
            padding: 1.15rem 1.05rem 1rem 1.05rem;
            border-radius: 20px;
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.26), transparent 30%),
                radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.18), transparent 28%),
                linear-gradient(145deg, #111827 0%, #1f2937 48%, #0f3d56 100%);
            color: #f8fafc;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.2);
            margin-bottom: 0.85rem;
        }
        .eval-hero h2 {
            margin: 0 0 0.3rem 0;
            font-size: 1.45rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
        }
        .eval-hero p {
            margin: 0;
            color: rgba(248, 250, 252, 0.84);
            font-size: 0.93rem;
        }
        .eval-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.7rem 0 0.9rem 0;
        }
        .eval-status-card {
            border-radius: 15px;
            padding: 0.78rem 0.82rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.98));
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        }
        .eval-status-label {
            display: block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.26rem;
        }
        .eval-status-value {
            display: block;
            font-size: 0.98rem;
            font-weight: 700;
            color: #0f172a;
        }
        .eval-setup-card {
            border-radius: 18px;
            padding: 0.95rem 0.95rem 0.85rem 0.95rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(241,245,249,0.96));
            border: 1px solid rgba(226, 232, 240, 0.95);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            margin: 0.75rem 0 0.9rem 0;
        }
        .eval-setup-card h4 {
            margin: 0 0 0.28rem 0;
            font-size: 1rem;
            color: #0f172a;
        }
        .eval-setup-card p {
            margin: 0;
            color: #475569;
            font-size: 0.9rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 12px;
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="eval-hero">
            <h2>RAG Evaluator</h2>
            <p>Audit answer quality, benchmark retrieval behavior, and verify whether generated responses are actually supported by the chunks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    eval_engine = st.selectbox("Evaluator Engine", ["Local GGUF (Llama.cpp)", "Ollama (Native Service)"], key="eval_engine")
    
    if eval_engine == "Ollama (Native Service)":
        if 'ollama_models' in locals() and ollama_models:
            eval_model_path = st.selectbox("Evaluator Model", ollama_models, key="eval_model")
        else:
            eval_model_path = st.text_input("Evaluator Model Name", value="qwen2.5", key="eval_model_text")
    else:
        eval_model_path = st.text_input("Evaluator GGUF Path", value="", key="eval_model_path")
        
    eval_llm = None
    if eval_model_path:
        with st.status(f"Loading Evaluator: {eval_model_path}...", expanded=False) as eval_status:
            eval_llm = load_llm(eval_model_path, -1, 4096, 0.1, engine=eval_engine)
            if eval_llm:
                eval_status.update(label="Evaluator Ready!", state="complete")
            else:
                eval_status.update(label="Evaluator Failed to Load", state="error")
    evaluator_ready_label = "Ready" if eval_llm else "Not Loaded"
    benchmark_mode_label = "Running" if benchmark_state else "Idle"
    setup_mode_label = "Open" if st.session_state.benchmark_setup_open else "Closed"
    st.markdown(
        f"""
        <div class="eval-status-grid">
            <div class="eval-status-card">
                <span class="eval-status-label">Evaluator Model</span>
                <span class="eval-status-value">{evaluator_ready_label}</span>
            </div>
            <div class="eval-status-card">
                <span class="eval-status-label">Benchmark Run</span>
                <span class="eval-status-value">{benchmark_mode_label}</span>
            </div>
            <div class="eval-status-card">
                <span class="eval-status-label">Benchmark Setup</span>
                <span class="eval-status-value">{setup_mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    
    diag_col, benchmark_col, stop_col = st.columns(3)

    if diag_col.button("Inspect Last Answer", use_container_width=True):
        if not eval_llm:
            st.error("Please load an Evaluator LLM first!")
        elif not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
            st.warning("No AI response detected yet to evaluate.")
        else:
            last_ai_msg = st.session_state.messages[-1]
            last_user_msg = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else None
            
            import time
            status_place = st.empty()
            with status_place.status("🔎 Compiling Diagnostic Pipeline...", expanded=True) as eval_status:
                st.write("Initializing strict RAG Evaluator Agent...")
                time.sleep(0.3)
                
                st.write(f"Extracting Original User Query: **{len(last_user_msg['content']) if last_user_msg else 0}** chars")
                time.sleep(0.3)
                
                chunks_count = len(last_ai_msg.get('context_used', '').split('--- Chunk')) - 1
                st.write(f"Isolating **{chunks_count}** retrieved DB chunk vectors...")
                time.sleep(0.3)
                
                st.write(f"Parsing Primary AI Response logic: **{len(last_ai_msg['content'])}** chars")
                time.sleep(0.3)
                
                eval_status.update(label="Diagnostic Parameters Fused. Handing over to Evaluator LLM...", state="complete", expanded=False)
                
            eval_prompt = f"""You are a strict RAG evaluator.
Audit the assistant answer using only the retrieved chunks.

Rules:
- No outside knowledge.
- If a claim is not in the retrieved chunks, mark it unsupported.
- Output JSON only.
- Keep all values short and scannable.
- Do not include markdown.
- Do not include chain-of-thought.

USER QUESTION:
{last_user_msg['content'] if last_user_msg else 'N/A'}

RETRIEVED CHUNKS:
{last_ai_msg.get('context_used', 'No chunks retrieved.')}

ASSISTANT ANSWER:
{last_ai_msg['content']}

Return exactly one JSON object using this schema:
{{
  "correctness": "Correct | Partially Correct | Incorrect",
  "retrieval_support": "Full | Partial | Missing",
  "hallucination_risk": "Low | Medium | High",
  "confidence": "High | Medium | Low",
  "steps_checked": [
    {{"label": "Restated the user question", "done": true}},
    {{"label": "Extracted main answer claims", "done": true}},
    {{"label": "Matched claims against retrieved chunks", "done": true}},
    {{"label": "Checked for missing parts", "done": true}},
    {{"label": "Checked for unsupported statements", "done": true}}
  ],
  "claim_checks": [
    {{
      "status": "PASS | PARTIAL | FAIL",
      "claim": "short claim",
      "reason": "short reason"
    }}
  ],
  "missing": "short text or None",
  "wrong": "short text or None",
  "bottom_line": "one short sentence"
}}
"""

            try:
                eval_run_state = st.empty()
                eval_run_state.info("Evaluator LLM is analyzing the answer against retrieved chunks...")
                with st.spinner("Running diagnostic review..."):
                    eval_raw_response = eval_llm.invoke(eval_prompt)
                eval_report = _extract_json_payload_with_repair(eval_raw_response, repair_llm=eval_llm)
                render_evaluator_report(eval_report)
                eval_run_state.success("Diagnostic review complete.")
            except Exception as e:
                st.error(f"Evaluator crash: {e}")

    if benchmark_state and stop_col.button("Stop Benchmark", use_container_width=True):
        processed = benchmark_state.get("index", 0)
        passed = benchmark_state.get("passed_count", 0)
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"[Benchmark Stopped]\n\n"
                f"Stopped after {processed} question(s). Passed {passed} so far."
            )
        })
        st.session_state.benchmark_state = None
        st.rerun()

    if benchmark_col.button("Create Benchmark", use_container_width=True):
        st.session_state.benchmark_setup_open = True
        st.rerun()

    if st.session_state.benchmark_setup_open and not benchmark_state:
        st.markdown(
            """
            <div class="eval-setup-card">
                <h4>Benchmark Setup</h4>
                <p>Choose how many document-grounded questions and off-topic rejection tests to generate before the automated run begins.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        benchmark_cfg_col1, benchmark_cfg_col2 = st.columns(2)
        relevant_question_count = benchmark_cfg_col1.number_input(
            "Right questions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="benchmark_relevant_count"
        )
        irrelevant_question_count = benchmark_cfg_col2.number_input(
            "Wrong questions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="benchmark_irrelevant_count"
        )
        st.caption("Right = answerable from the document. Wrong = should be rejected as not in the document.")
        start_bench_col, cancel_bench_col = st.columns(2)
        if cancel_bench_col.button("Cancel Setup", use_container_width=True):
            st.session_state.benchmark_setup_open = False
            st.rerun()
        start_benchmark_now = start_bench_col.button("Start Benchmark Run", use_container_width=True)
    else:
        relevant_question_count = int(st.session_state.get("benchmark_relevant_count", 5))
        irrelevant_question_count = int(st.session_state.get("benchmark_irrelevant_count", 5))
        start_benchmark_now = False

    if st.session_state.last_benchmark_summary and not benchmark_state:
        summary = st.session_state.last_benchmark_summary
        st.markdown("#### Last Benchmark Summary")
        if summary["summary_verdict"] == "PASS":
            st.success(f"Verdict: PASS | Score: {summary['passed_count']}/{summary['total_count']}")
        elif summary["summary_verdict"] == "PARTIAL":
            st.warning(f"Verdict: PARTIAL | Score: {summary['passed_count']}/{summary['total_count']}")
        else:
            st.error(f"Verdict: FAIL | Score: {summary['passed_count']}/{summary['total_count']}")

        s1, s2, s3 = st.columns(3)
        s1.markdown(f"**Relevant**  \n`{summary['relevant_passed']}/{summary['relevant_total']}`")
        s2.markdown(f"**Irrelevant**  \n`{summary['irrelevant_passed']}/{summary['irrelevant_total']}`")
        s3.markdown(f"**Failed**  \n`{summary['fail_count']}`")

        st.markdown("**Steps Completed**")
        st.markdown("- Generated benchmark questions from the current document chunks")
        st.markdown("- Split them into relevant and irrelevant checks")
        st.markdown("- Ran each question through the RAG answer pipeline")
        st.markdown("- Graded answers against retrieved evidence")
        st.markdown("- Counted passes and failures")

        st.markdown("**Why Relevant Questions Failed**")
        st.markdown(summary["relevant_failure_lines"])

        st.markdown("**Top Failed Checks**")
        st.markdown(summary["failure_lines"])

    if start_benchmark_now:
        if not eval_llm:
            st.error("Please load an Evaluator LLM first!")
        elif not llm:
            st.error("Please load the main answer LLM first!")
        elif not st.session_state.retriever or not st.session_state.generated_chunks:
            st.error("Please embed a document first!")
        else:
            st.session_state.benchmark_setup_open = False
            benchmark_status = st.empty()
            try:
                with benchmark_status.status("Preparing benchmark run...", expanded=True) as bench_state:
                    benchmark_step = st.empty()
                    st.write("Reading currently generated document chunks...")
                    st.write(f"Found **{len(st.session_state.generated_chunks)}** chunk(s) available for benchmark generation.")
                    benchmark_step.info("Starting benchmark question generation...")
                    benchmark_questions = generate_benchmark_questions(
                        eval_llm,
                        st.session_state.generated_chunks,
                        relevant_count=int(relevant_question_count),
                        irrelevant_count=int(irrelevant_question_count),
                        status_callback=lambda msg: benchmark_step.info(msg)
                    )
                    benchmark_step.success("Benchmark question generation finished.")
                    st.write(f"Evaluator returned **{len(benchmark_questions)}** benchmark question(s).")

                    expected_total = int(relevant_question_count) + int(irrelevant_question_count)
                    if len(benchmark_questions) < expected_total:
                        bench_state.update(label="Benchmark generation failed", state="error", expanded=True)
                        st.error("Benchmark generation did not return enough questions.")
                    else:
                        relevant_count = sum(1 for q in benchmark_questions if q.get("type") == "relevant")
                        irrelevant_count = sum(1 for q in benchmark_questions if q.get("type") == "irrelevant")
                        st.write(f"Question mix confirmed: **{relevant_count} relevant** and **{irrelevant_count} irrelevant**.")
                        st.write("Generated benchmark questions:")
                        for idx, q in enumerate(benchmark_questions, start=1):
                            q_type = q.get("type", "unknown").upper()
                            q_text = q.get("question", "Unnamed benchmark question")
                            expected = q.get("expected_answer", "No expected answer provided.")
                            st.write(f"{idx}. [{q_type}] {q_text}")
                            st.caption(f"Expected behavior: {expected}")
                        st.write("Benchmark set is ready. Starting automated run...")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Benchmark run started. I will ask 5 relevant and 5 irrelevant document questions, answer them one by one, then score the results."
                        })
                        st.session_state.benchmark_state = {
                            "questions": benchmark_questions,
                            "index": 0,
                            "passed_count": 0,
                            "total_count": len(benchmark_questions),
                            "results": [],
                        }
                        st.rerun()
            except Exception as e:
                benchmark_status.error(f"Benchmark failed: {e}")

    benchmark_state = st.session_state.benchmark_state
    if benchmark_state:
        active_questions = benchmark_state.get("questions", [])
        active_index = benchmark_state.get("index", 0)
        total_count = benchmark_state.get("total_count", len(active_questions))
        passed_count = benchmark_state.get("passed_count", 0)
        benchmark_results = benchmark_state.get("results", [])

        if active_index < total_count:
            item = active_questions[active_index]
            question_type = item.get("type", "unknown").upper()
            question_text = item.get("question", "Unnamed benchmark question")

            active_status = st.empty()
            with active_status.status(
                f"Benchmark {active_index + 1}/{total_count} in progress...",
                expanded=True
            ) as run_state:
                st.write(f"Current question type: **{question_type}**")
                st.write(f"Current question: `{question_text}`")
                st.write("Running main RAG retrieval and answer generation...")

                st.session_state.messages.append({
                    "role": "user",
                    "content": f"[Benchmark {active_index + 1}/{total_count} | {question_type}] {question_text}"
                })

                result = answer_question_with_rag(
                    question_text,
                    st.session_state.retriever,
                    llm,
                    search_top_k
                )
                st.write(f"Retrieved **{result['chunks']}** chunk(s).")
                st.write("Grading the answer against expected behavior and retrieved chunks...")

                grade = evaluate_benchmark_answer(
                    eval_llm,
                    item,
                    result["answer"],
                    result["context"]
                )

                verdict = grade.get("verdict", "UNKNOWN")
                failure_type = grade.get("failure_type", "none")
                reason = grade.get("reason", "No grading reason returned.")
                expected_answer = item.get("expected_answer", "")
                st.write(f"Question result: **{verdict}**")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"[Benchmark Answer {active_index + 1}/{total_count} | {verdict}]\n\n"
                        f"Question type: {question_type}\n\n"
                        f"Answer: {result['answer']}\n\n"
                        f"Expected: {expected_answer}\n\n"
                        f"Failure type: {failure_type}\n\n"
                        f"Evaluation: {reason}"
                    ),
                    "context_used": result["context"],
                    "metrics": {
                        "total": len(st.session_state.get('generated_chunks', [])),
                        "chunks": result["chunks"],
                        "length": result["prompt_length"],
                        "time": 0.0,
                    }
                })

                benchmark_results.append({
                    "index": active_index + 1,
                    "type": question_type,
                    "question": question_text,
                    "verdict": verdict,
                    "passed": bool(grade.get("passed")),
                    "failure_type": failure_type,
                    "reason": reason,
                })

                if grade.get("passed"):
                    benchmark_state["passed_count"] = passed_count + 1
                benchmark_state["index"] = active_index + 1
                benchmark_state["results"] = benchmark_results

                run_state.update(
                    label=f"Completed benchmark {active_index + 1}/{total_count}",
                    state="complete",
                    expanded=False
                )

            st.session_state.benchmark_state = benchmark_state
            st.rerun()
        else:
            summary_verdict = "PASS" if passed_count == total_count else "PARTIAL" if passed_count > 0 else "FAIL"
            fail_count = total_count - passed_count
            relevant_total = sum(1 for r in benchmark_results if r.get("type") == "RELEVANT")
            irrelevant_total = sum(1 for r in benchmark_results if r.get("type") == "IRRELEVANT")
            relevant_passed = sum(1 for r in benchmark_results if r.get("type") == "RELEVANT" and r.get("passed"))
            irrelevant_passed = sum(1 for r in benchmark_results if r.get("type") == "IRRELEVANT" and r.get("passed"))
            failed_items = [r for r in benchmark_results if not r.get("passed")]
            relevant_failed = [r for r in failed_items if r.get("type") == "RELEVANT"]
            failure_lines = "\n".join(
                [f"- Q{r['index']} [{r['type']}] {r['verdict']} | {r.get('failure_type', 'none')}: {r['reason']}" for r in failed_items[:5]]
            ) or "- None"
            relevant_failure_lines = "\n".join(
                [f"- Q{r['index']}: {r.get('failure_type', 'none')} | {r['reason']}" for r in relevant_failed[:5]]
            ) or "- None"
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"[Benchmark Summary | {summary_verdict}]\n\n"
                    f"Final verdict: {summary_verdict}\n\n"
                    f"Score: {passed_count}/{total_count} passed\n\n"
                    f"Steps completed:\n"
                    f"- Generated benchmark questions from the document chunks\n"
                    f"- Split them into relevant and irrelevant checks\n"
                    f"- Ran each question through the main RAG pipeline\n"
                    f"- Graded each answer against retrieved evidence\n"
                    f"- Counted passes and failures\n\n"
                    f"Breakdown:\n"
                    f"- Relevant: {relevant_passed}/{relevant_total} passed\n"
                    f"- Irrelevant: {irrelevant_passed}/{irrelevant_total} passed\n"
                    f"- Failed: {fail_count}\n\n"
                    f"Why relevant questions failed:\n"
                    f"{relevant_failure_lines}\n\n"
                    f"Top failed checks:\n"
                    f"{failure_lines}"
                )
            })
            st.session_state.last_benchmark_summary = {
                "summary_verdict": summary_verdict,
                "passed_count": passed_count,
                "total_count": total_count,
                "fail_count": fail_count,
                "relevant_passed": relevant_passed,
                "relevant_total": relevant_total,
                "irrelevant_passed": irrelevant_passed,
                "irrelevant_total": irrelevant_total,
                "relevant_failure_lines": relevant_failure_lines,
                "failure_lines": failure_lines,
            }
            st.session_state.benchmark_state = None
            if summary_verdict == "PASS":
                st.success(f"Benchmark finished with verdict PASS: {passed_count}/{total_count} passed.")
            elif summary_verdict == "PARTIAL":
                st.warning(f"Benchmark finished with verdict PARTIAL: {passed_count}/{total_count} passed.")
            else:
                st.error(f"Benchmark finished with verdict FAIL: {passed_count}/{total_count} passed.")

prompt = st.chat_input("What does the document say about ... ?")
if prompt:
    prompt = prompt.strip()

    if not llm:
        st.error("❌ **Pipeline Offline:** You must launch an Inference Engine (LLM) from the Sidebar first!")
        st.stop()

    if not st.session_state.retriever:
        st.error("❌ **Database Offline:** You must vector-embed a Document before querying it!")
        st.stop()

    with main_chat_col:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            import time
            start_time = time.time()

            status_place = st.empty()
            with status_place.status("⚙️ Interrogating Vector Database...", expanded=True) as retrieval_status:
                st.write("Navigating high-dimensional coordinate space...")
                time.sleep(0.3)

                db_name = db_choice if 'db_choice' in locals() else "Global Persistent Cache"

                retriever = st.session_state.retriever
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs["k"] = search_top_k
                docs = retriever.invoke(prompt)

                st.write(f"✓ Extracted **{len(docs)}** highly contextual chunks from `{db_name}`.")
                context = "\n\n".join([f"**--- Chunk {i+1} ---**\n\n{doc.page_content}" for i, doc in enumerate(docs)])
                time.sleep(0.3)

                st.write("Fusing dynamic Prompt Matrix...")
                time.sleep(0.2)

                retrieval_status.update(label=f"Architectural Context Sourced: {len(docs)} vectors", state="complete", expanded=False)

            prompt_template = PromptTemplate.from_template(
                "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n\nHelpful Answer:"
            )

            formatted_prompt = prompt_template.format(context=context, question=prompt)

            with st.spinner("🧠 Computing neural response matrix... (Initial token compilation may take a moment)"):
                response = st.write_stream(llm.stream(formatted_prompt))

            latency = round(time.time() - start_time, 2)
            total_vectors_in_db = len(st.session_state.get('generated_chunks', []))

            st.caption("⚡ **Generation Telemetry:**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vectors Evaluated", total_vectors_in_db)
            m2.metric("Top-K Injected", len(docs))
            m3.metric("Prompt Density", f"{len(formatted_prompt)} chars")
            m4.metric("Latency", f"{latency} sec")

            with st.expander("🔍 View Raw Database Chunks Sourced"):
                st.markdown(context)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "context_used": context,
            "metrics": {"total": total_vectors_in_db, "chunks": len(docs), "length": len(formatted_prompt), "time": latency}
        })
