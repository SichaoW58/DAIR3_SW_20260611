"""
cls_rag.py
Per-agent retrieval-augmented generation: store, retrieve, and assemble.

One ``KnowledgeBase`` per agent. Each lives at:

    FOO/knowledge/<AgentName>/
        sources/               <-- raw documents (PDF, .md, .txt, .html, ...)
        .index/                <-- ChromaDB persistent client + manifest
        index_manifest.json    <-- which embedding backend was used, file mtimes

The embedding backend is chosen at first index time and persisted in the
manifest. Changing it later forces a full re-index because the dimensionality
won't match the existing collection.

Supported embedding backends:

    'openai'    text-embedding-3-small   (cloud, ~$0.02/M tokens)
    'ollama'    nomic-embed-text         (local, free, requires Ollama)

Chunking: token-naive paragraph splitter with a soft maximum word count
(default 350 words ~ ~500 tokens) and a one-paragraph overlap. PDFs go through
PyMuPDF; everything else is read as UTF-8 text.

Retrieval returns a list of ``Chunk`` dicts with ``text``, ``source`` (file
path relative to ``sources/``), ``chunk_id``, and ``score``. The GUI uses
``source`` + ``chunk_id`` to render citations next to the answer.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import os
import re
import json
import hashlib
from datetime import datetime


# ----- Consent + provenance: ethics hooks (see slides_7.3) -----------------

class ConsentRequiredError(RuntimeError):
    """Raised by ``KnowledgeBase.ingest_file`` when no consent has been
    recorded for the currently-selected embedding backend. The GUI catches
    this, displays ``consent_text(backend)``, records the user's
    affirmation via ``record_consent``, and retries the ingest.

    Keeping the gate inside ``ingest_file`` (rather than only in the GUI)
    means the policy is enforced even if a future caller bypasses the UI.
    """


def consent_text(backend):
    """The disclosure text shown in the consent gate. Backend-specific
    because the data-egress posture is the entire ethical question."""
    if backend == "openai":
        return (
            "OpenAI embedding backend selected.\n\n"
            "By indexing material into this knowledge base, you affirm:\n\n"
            "  (1) You have the right to embed this material under this backend.\n"
            "  (2) Each chunk of every file you ingest WILL BE TRANSMITTED to OpenAI.\n"
            "  (3) NIH peer reviewers are PROHIBITED (NIH NOT-OD-23-149, June 2023)\n"
            "      from using generative AI on grant applications they review.\n"
            "      Do not ingest proposal text for which you are a peer reviewer.\n"
            "  (4) NSF reviewers are PROHIBITED (NSF, June 2023) from uploading\n"
            "      proposal content to non-approved AI tools.\n"
            "  (5) Material containing PHI, unpublished work by others, or material\n"
            "      under embargo should be embedded only with a backend approved by\n"
            "      your institution's data-handling policy.\n\n"
            "If any of the above is uncertain, cancel and use the Ollama (local)\n"
            "backend, which keeps chunks on this machine."
        )
    if backend == "ollama":
        return (
            "Ollama (local) embedding backend selected.\n\n"
            "By indexing material into this knowledge base, you affirm:\n\n"
            "  (1) You have the right to embed this material under this backend.\n"
            "  (2) Embeddings are computed by the local Ollama daemon and stored\n"
            "      ONLY on this machine. Chunks are NOT transmitted to a vendor.\n"
            "  (3) The federal-agency prohibitions on AI in peer review still apply\n"
            "      (NIH NOT-OD-23-149; NSF June 2023). Local computation does not\n"
            "      exempt you from those rules.\n\n"
            "Local embedding mitigates data egress; it does not waive consent\n"
            "or policy obligations of the source authors or your institution."
        )
    return f"Embedding backend '{backend}' selected. Affirm right to embed."


def extract_provenance(file_path):
    """Extract author / title / year metadata for a source file.

    Order of precedence:
      1. ``<filename>.meta.json`` sidecar (user-authored). Schema:
         ``{"author": str, "title": str, "year": str}``. Highest priority
         because the user knows things the file's embedded metadata may not.
      2. PDF embedded metadata (PyMuPDF: /Author, /Title, /CreationDate).
      3. Empty dict.

    Returns a flat dict with string-or-None values (Chroma metadata
    constraint). Citation rendering uses ``cite_key`` (last-name + year)
    when both author and year are present.
    """
    sidecar = file_path + ".meta.json"
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return {
                "author": obj.get("author"),
                "title": obj.get("title"),
                "year": str(obj["year"]) if obj.get("year") is not None else None,
            }
        except Exception:
            pass

    if file_path.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(file_path)
            meta = doc.metadata or {}
            doc.close()
            year = None
            cd = meta.get("creationDate") or meta.get("CreationDate") or ""
            m = re.search(r"(\d{4})", cd or "")
            if m:
                year = m.group(1)
            return {
                "author": (meta.get("author") or meta.get("Author") or None) or None,
                "title": (meta.get("title") or meta.get("Title") or None) or None,
                "year": year,
            }
        except Exception:
            return {}

    return {}


def make_cite_key(provenance):
    """Last-name + year (e.g. 'Smith 2023') when available, else None."""
    if not provenance:
        return None
    author = provenance.get("author")
    year = provenance.get("year")
    if not author:
        return None
    # Conservative last-name extraction: take the last token of the first author
    # (split on comma -> first author; split on whitespace -> last token).
    first_author = author.split(",")[0].strip()
    last_name = first_author.split()[-1] if first_author else None
    if not last_name:
        return None
    return f"{last_name} {year}" if year else last_name


def render_citation_tag(chunk):
    """Compose the bracketed tag for a retrieved chunk. Uses cite_key when
    present, otherwise falls back to source#chunk_id."""
    cite = (chunk.get("cite_key") or "").strip()
    base = f"{chunk['source']}#{chunk['chunk_id']}"
    return f"[{cite}, {base}]" if cite else f"[{base}]"


# ----- Embedding backends ---------------------------------------------------

OPENAI_EMBED_MODEL = "text-embedding-3-small"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def _embed_openai(texts, model=OPENAI_EMBED_MODEL):
    import openai
    client = openai.OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _embed_ollama(texts, model=OLLAMA_EMBED_MODEL):
    from cls_ollama import embed_with_ollama
    return embed_with_ollama(texts, model=model)


_BACKENDS = {
    "openai": (OPENAI_EMBED_MODEL, _embed_openai),
    "ollama": (OLLAMA_EMBED_MODEL, _embed_ollama),
}


class _ExternalEmbeddingsOnly:
    """Stub embedding function handed to every Chroma collection we create.

    Why this exists: Chroma's default ``embedding_function`` is
    ``SentenceTransformerEmbeddingFunction``, which loads
    ``all-MiniLM-L6-v2`` via ``onnxruntime`` on first use. On some Windows
    setups the DLL load fails non-gracefully and terminates the host
    Python process (the GUI just disappears with no traceback).

    This kb ALWAYS passes vectors explicitly through ``embeddings=...`` /
    ``query_embeddings=...``, so Chroma never needs an embedder. If it
    ever tries (caller bug), the RuntimeError below surfaces in the
    chat instead of crashing the process.
    """

    def __call__(self, input):
        raise RuntimeError(
            "Chroma tried to embed input itself; this knowledge base "
            "provides explicit vectors. Caller bug."
        )

    def name(self):
        return "external_only"


def available_backends():
    """Backends usable in the current environment. 'openai' needs an API key;
    'ollama' needs the daemon reachable."""
    out = []
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    try:
        import requests
        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        requests.get(f"{base}/api/tags", timeout=2)
        out.append("ollama")
    except Exception:
        pass
    return out


# ----- Chunking -------------------------------------------------------------

def _read_pdf(path):
    """Extract text from a PDF using PyMuPDF. One page per double newline."""
    import fitz
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n\n".join(parts)


def _read_text_any(path):
    """Read PDFs via PyMuPDF and everything else as UTF-8 text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def chunk_text(text, max_words=350, overlap_paragraphs=1):
    """Split text into paragraph-based chunks, each at most ``max_words``
    words. Paragraphs are separated by blank lines. Adjacent chunks overlap
    by ``overlap_paragraphs`` whole paragraphs to soften retrieval cliffs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    cur = []
    cur_words = 0
    for para in paragraphs:
        words = len(para.split())
        if cur and cur_words + words > max_words:
            chunks.append("\n\n".join(cur))
            # Start the next chunk with the last ``overlap_paragraphs`` paragraphs.
            cur = cur[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            cur_words = sum(len(p.split()) for p in cur)
        cur.append(para)
        cur_words += words
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ----- KnowledgeBase --------------------------------------------------------

def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _agent_root(agent_name):
    return os.path.join(_here(), "knowledge", agent_name)


def ensure_knowledge_dir(agent_name):
    """Create FOO/knowledge/<agent>/{sources,.index}/ if missing.
    Returns the (sources_dir, index_dir) pair."""
    root = _agent_root(agent_name)
    sources = os.path.join(root, "sources")
    index = os.path.join(root, ".index")
    os.makedirs(sources, exist_ok=True)
    os.makedirs(index, exist_ok=True)
    return sources, index


def _manifest_path(agent_name):
    return os.path.join(_agent_root(agent_name), "index_manifest.json")


def load_manifest(agent_name):
    """Return the index manifest dict, or {} if none exists yet."""
    p = _manifest_path(agent_name)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(agent_name, manifest):
    with open(_manifest_path(agent_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def consent_status(agent_name, backend):
    """Return the persisted consent record for (agent, backend) or None.
    Truthy means consent has been recorded; the dict carries a timestamp."""
    m = load_manifest(agent_name)
    return (m.get("consent") or {}).get(backend)


def record_consent(agent_name, backend):
    """Persist that consent was given for (agent, backend) at this moment.
    Re-callable; updates the timestamp. Ensures the agent directory exists
    so it can be called standalone (e.g. from the consent dialog) before a
    KnowledgeBase has been instantiated."""
    ensure_knowledge_dir(agent_name)
    m = load_manifest(agent_name)
    consent = m.setdefault("consent", {})
    consent[backend] = {
        "given_at": datetime.now().isoformat(),
        "backend": backend,
    }
    save_manifest(agent_name, m)


def revoke_consent(agent_name, backend):
    """Remove any consent record for (agent, backend). The next ingest will
    re-prompt. Use after a backend switch or a policy review."""
    ensure_knowledge_dir(agent_name)
    m = load_manifest(agent_name)
    consent = m.setdefault("consent", {})
    if backend in consent:
        del consent[backend]
        save_manifest(agent_name, m)


class KnowledgeBase:
    """Per-agent retrieval store. Lazy: opens its Chroma collection on first
    use; no daemon, no network calls until the user actually indexes or queries.

    The agent name is the partitioning key. Two agents with the same model
    but different personas have separate knowledge bases. This is intentional
    — see slides_7.3 for the pedagogical argument.
    """

    def __init__(self, agent_name, backend=None):
        self.agent_name = agent_name
        self.sources_dir, self.index_dir = ensure_knowledge_dir(agent_name)
        self.manifest = load_manifest(agent_name)

        # Backend: prefer the persisted choice; else honored arg; else None.
        # None means "not yet decided"; ingest_file() will require an explicit choice.
        self.backend = backend or self.manifest.get("backend")

        self._client = None
        self._collection = None

    # --- backend management -------------------------------------------------

    def set_backend(self, backend):
        """Persist the chosen backend. Wipes the index if changed (embedding
        dimensions differ across backends)."""
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown embedding backend: {backend!r}")
        if self.manifest.get("backend") and self.manifest["backend"] != backend:
            self.wipe_index()
        self.backend = backend
        self.manifest["backend"] = backend
        self.manifest["backend_model"] = _BACKENDS[backend][0]
        save_manifest(self.agent_name, self.manifest)

    def backend_label(self):
        """Short human-readable string for the chat header."""
        if not self.backend:
            return "RAG: not configured"
        provider = "OpenAI (cloud)" if self.backend == "openai" else "Ollama (local)"
        return f"RAG: {provider} - {_BACKENDS[self.backend][0]}"

    # --- chroma plumbing ----------------------------------------------------

    def _open_collection(self):
        if self._collection is not None:
            return self._collection
        print(f"[rag:{self.agent_name}] _open_collection: importing chromadb", flush=True)
        import chromadb
        print(f"[rag:{self.agent_name}] _open_collection: PersistentClient(path={self.index_dir})", flush=True)
        self._client = chromadb.PersistentClient(path=self.index_dir)
        coll_name = f"kb_{self.backend or 'unset'}"
        print(f"[rag:{self.agent_name}] _open_collection: get_or_create_collection({coll_name!r}) with stub embed fn", flush=True)
        # Always pass a no-op embedding function so Chroma does not load
        # its default ONNX-based embedder (see _ExternalEmbeddingsOnly).
        # Older Chroma versions that reject arbitrary callables fall back
        # to the default; the one-time ONNX load is the cost in that case.
        try:
            self._collection = self._client.get_or_create_collection(
                name=coll_name,
                embedding_function=_ExternalEmbeddingsOnly(),
            )
        except Exception as e:
            print(f"[rag:{self.agent_name}] _open_collection: stub embed fn rejected ({type(e).__name__}: {e}); retrying with default", flush=True)
            self._collection = self._client.get_or_create_collection(name=coll_name)
        print(f"[rag:{self.agent_name}] _open_collection: collection ready, count={self._collection.count()}", flush=True)
        return self._collection

    def _embed(self, texts):
        if not self.backend:
            raise RuntimeError(
                "No embedding backend chosen. Call set_backend('openai') or "
                "set_backend('ollama') first."
            )
        _, fn = _BACKENDS[self.backend]
        return fn(texts)

    # --- ingestion ----------------------------------------------------------

    def wipe_index(self):
        """Delete the on-disk Chroma index and reset the manifest's file list."""
        import shutil
        if os.path.isdir(self.index_dir):
            shutil.rmtree(self.index_dir, ignore_errors=True)
        os.makedirs(self.index_dir, exist_ok=True)
        self._client = None
        self._collection = None
        self.manifest["files"] = {}
        save_manifest(self.agent_name, self.manifest)

    def ingest_file(self, file_path, status_callback=None):
        """Chunk + embed + store a single file. If the file is already up to
        date (same mtime as recorded in the manifest), no-op.

        Raises ``ConsentRequiredError`` if no consent has been recorded for
        the active embedding backend. The router catches and presents the
        consent dialog, calls ``record_consent``, then retries.
        """

        def _emit(msg):
            if status_callback:
                status_callback(msg)

        # Enforce the consent gate before any data leaves the local process.
        if not self.backend:
            raise RuntimeError(
                "No embedding backend selected. Open the RAG settings and pick one."
            )
        if not consent_status(self.agent_name, self.backend):
            raise ConsentRequiredError(self.backend)

        # Copy into sources/ if it lives elsewhere (so the kb is self-contained).
        filename = os.path.basename(file_path)
        dest = os.path.join(self.sources_dir, filename)
        print(f"[rag:{self.agent_name}] ingest: filename={filename}", flush=True)
        if os.path.abspath(file_path) != os.path.abspath(dest):
            _emit(f"Copying {filename} into sources/")
            import shutil
            shutil.copy2(file_path, dest)
            # Carry the sidecar over too if present, so provenance survives.
            side = file_path + ".meta.json"
            if os.path.isfile(side):
                shutil.copy2(side, dest + ".meta.json")

        mtime = os.path.getmtime(dest)
        files = self.manifest.setdefault("files", {})
        if files.get(filename, {}).get("mtime") == mtime:
            print(f"[rag:{self.agent_name}] ingest: already indexed (mtime match); skipping", flush=True)
            _emit(f"{filename} already indexed; skipping")
            return 0

        print(f"[rag:{self.agent_name}] ingest: reading text", flush=True)
        _emit(f"Reading {filename}")
        text = _read_text_any(dest)
        print(f"[rag:{self.agent_name}] ingest: text length = {len(text)}", flush=True)
        _emit(f"Chunking {filename}")
        chunks = chunk_text(text)
        print(f"[rag:{self.agent_name}] ingest: {len(chunks)} chunks produced", flush=True)
        if not chunks:
            _emit(f"{filename}: no extractable text")
            return 0

        print(f"[rag:{self.agent_name}] ingest: extracting provenance", flush=True)
        _emit(f"Extracting provenance metadata for {filename}")
        prov = extract_provenance(dest)
        cite_key = make_cite_key(prov) or ""
        print(f"[rag:{self.agent_name}] ingest: provenance={prov}, cite_key={cite_key!r}", flush=True)

        print(f"[rag:{self.agent_name}] ingest: embedding via {self.backend} (this can take a while)", flush=True)
        _emit(f"Embedding {len(chunks)} chunk(s) via {self.backend}")
        vectors = self._embed(chunks)
        print(f"[rag:{self.agent_name}] ingest: got {len(vectors)} embedding vectors, dim={len(vectors[0]) if vectors else 'NA'}", flush=True)

        print(f"[rag:{self.agent_name}] ingest: opening Chroma collection", flush=True)
        coll = self._open_collection()
        print(f"[rag:{self.agent_name}] ingest: clearing prior chunks for {filename}", flush=True)
        try:
            coll.delete(where={"source": filename})
        except Exception as e:
            print(f"[rag:{self.agent_name}] ingest: delete-prior failed ({type(e).__name__}: {e}); continuing", flush=True)

        ids, metadatas, documents = [], [], []
        for i, chunk in enumerate(chunks):
            cid = hashlib.sha1(f"{filename}::{i}::{mtime}".encode()).hexdigest()[:16]
            ids.append(cid)
            # Chroma metadata values must be primitive (str / int / float / bool).
            # None is not allowed, so we coerce missing provenance to "".
            metadatas.append({
                "source": filename,
                "chunk_id": i,
                "cite_key": cite_key or "",
                "author": prov.get("author") or "",
                "title": prov.get("title") or "",
                "year": prov.get("year") or "",
            })
            documents.append(chunk)
        print(f"[rag:{self.agent_name}] ingest: adding {len(ids)} rows to Chroma", flush=True)
        coll.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
        print(f"[rag:{self.agent_name}] ingest: persisted; updating manifest", flush=True)

        files[filename] = {
            "mtime": mtime,
            "chunks": len(chunks),
            "indexed_at": datetime.now().isoformat(),
            "provenance": prov,
            "cite_key": cite_key,
        }
        save_manifest(self.agent_name, self.manifest)
        print(f"[rag:{self.agent_name}] ingest: DONE", flush=True)
        return len(chunks)

    def ingest_all_sources(self, status_callback=None):
        """Walk sources/ and ingest anything new or modified."""
        total = 0
        for name in sorted(os.listdir(self.sources_dir)):
            path = os.path.join(self.sources_dir, name)
            if os.path.isfile(path):
                total += self.ingest_file(path, status_callback=status_callback)
        return total

    # --- retrieval ----------------------------------------------------------

    def query(self, question, top_k=4):
        """Return the top_k most relevant chunks as a list of dicts.
        Empty list if the index is empty or no backend is set."""
        if not self.backend:
            return []
        try:
            qvec = self._embed([question])[0]
        except Exception as e:
            print(f"RAG embed failed: {e}")
            return []

        coll = self._open_collection()
        if coll.count() == 0:
            return []

        results = coll.query(
            query_embeddings=[qvec],
            n_results=min(top_k, coll.count()),
        )
        out = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            m = meta or {}
            out.append({
                "text": doc,
                "source": m.get("source", "?"),
                "chunk_id": m.get("chunk_id", 0),
                "cite_key": m.get("cite_key", "") or "",
                "author": m.get("author", "") or "",
                "title": m.get("title", "") or "",
                "year": m.get("year", "") or "",
                "score": dist,
            })
        return out

    def count(self):
        """Live chunk count from the Chroma index. Opens Chroma if needed.
        For startup / status-line display use ``manifest_chunk_count``
        instead — it does not touch Chroma."""
        if not self.backend:
            return 0
        try:
            return self._open_collection().count()
        except Exception:
            return 0

    def manifest_chunk_count(self):
        """Sum of chunk counts recorded in the manifest. Pure file I/O;
        does NOT initialize Chroma. Use this on GUI startup paths so a
        fragile Chroma/ONNX/SQLite DLL load can't terminate the process
        before the window even appears.

        May diverge from ``count()`` if the manifest and the Chroma
        index get out of sync — the manifest is the source of truth for
        what *we* ingested, the index is what Chroma actually holds.
        """
        total = 0
        for entry in (self.manifest.get("files") or {}).values():
            try:
                total += int(entry.get("chunks", 0))
            except (TypeError, ValueError):
                pass
        return total

    def manifest_source_count(self):
        """Number of source files recorded in the manifest. Pure file I/O."""
        return len(self.manifest.get("files") or {})


# ----- Prompt assembly ------------------------------------------------------

def build_rag_prompt(question, chunks, prelude=None):
    """Compose a single user message that bundles the question with retrieved
    context. The prompt asks the model to cite sources by their bracketed
    tag — when provenance metadata is present the tag is
    ``[Author Year, source#chunk]``; otherwise just ``[source#chunk]``.
    """
    if not chunks:
        return question
    header = prelude or (
        "Answer the question using the context excerpts below. Cite each "
        "claim with the bracketed tag shown above its excerpt. If the context "
        "does not contain the answer, say so explicitly rather than guessing."
    )
    blocks = []
    for c in chunks:
        tag = render_citation_tag(c)
        blocks.append(f"{tag}\n{c['text']}")
    return f"{header}\n\n=== CONTEXT ===\n\n" + "\n\n---\n\n".join(blocks) + f"\n\n=== QUESTION ===\n\n{question}"


def render_citations(chunks):
    """Pretty-print citations for the chat log. Surfaces author/year when
    extracted from PDF metadata or a sidecar ``.meta.json``."""
    if not chunks:
        return ""
    lines = ["**Retrieved context:**"]
    for c in chunks:
        snippet = c["text"].replace("\n", " ")[:140]
        tag = render_citation_tag(c)
        lines.append(f"- `{tag}` (score={c['score']:.3f}) {snippet}...")
    return "\n".join(lines)
