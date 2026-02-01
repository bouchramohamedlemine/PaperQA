# Chunk learning materials by section/subsection (LlamaIndex SentenceSplitter), then embed with OpenAI text-embedding-3.
import json
import os
import re
from typing import List, Optional, Tuple

from llama_index.core import Document as LlamaIxDocument
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
from pypdf import PdfReader


from Preprocessing.ChunkData import Document, RAGChunk

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class Document_processing:
    def __init__(self, documents_paths, supabase_client):
        self.documents = documents_paths
        self.supabase_client = supabase_client
        self.sentence_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )



    def _arxiv_id_from_path(self, doc_path: str) -> str:
        """Arxiv id is the document filename without extension (e.g. 2005.11401v4)."""
        return os.path.splitext(os.path.basename(doc_path))[0]



    def _extract_title_and_abstract_from_pdf(self, doc_path: str) -> Tuple[str, str]:
        """Extract title and abstract from the first page. Returns (title, abstract)."""
        reader = PdfReader(doc_path)
        first_page_text = (reader.pages[0].extract_text() or "").strip()
        if not first_page_text:
            return "", ""

        lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
        if not lines:
            return "", ""

        def _looks_like_author_line(line: str) -> bool:
            lower = line.lower()
            if "@" in line or ".com" in lower:
                return True
            if any(c in line for c in "†‡⋆"):
                return True
            if "university" in lower or "research" in lower or "institute" in lower:
                return True
            if line.count(",") >= 2:
                return True
            return False

        title_parts = []
        for ln in lines:
            if re.match(r"^\s*abstract\s*$", ln, re.IGNORECASE):
                break
            if _looks_like_author_line(ln):
                break
            title_parts.append(ln)
        title = " ".join(title_parts).strip() if title_parts else lines[0]

        abstract_lines = []
        for i, line in enumerate(lines):
            if re.match(r"^\s*abstract\s*$", line, re.IGNORECASE):
                for ln in lines[i + 1 :]:
                    if re.match(r"^\s*\d+\.?\s", ln):
                        break
                    if re.match(r"^\s*(introduction|keywords|index terms)\s*$", ln, re.IGNORECASE):
                        break
                    if ln.strip().lower().startswith("arxiv:"):
                        break
                    abstract_lines.append(ln)
                break
        abstract = " ".join(abstract_lines).strip()
        return title, abstract




    def _extract_section_usage_line(self, section_text: str) -> str:
        """One sentence per section: what this paper used/proposed/found in this section."""
        if not section_text or not section_text.strip():
            return ""
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are given a section of an academic paper. Write exactly ONE short line (one sentence) "
                        "that states what was used, proposed, or found in THIS section. Keep it as short and concise "
                        "as possible, no filler words. Include key terms: model names, dataset names, method names, "
                        "metrics, or tools. Mention only what this paper did, not prior or related work. Do NOT start "
                        "with 'This paper' or 'The paper'; start directly with the fact, method, or result. The line "
                        "will be used for search, so pack in the important keywords. Reply with only that one line, "
                        "nothing else."
                    ),
                },
                {"role": "user", "content": f"Section text:\n{section_text[:8000]}"},
            ],
            temperature=0,
        )
        line = (resp.choices[0].message.content or "").strip()
        return line.split("\n")[0].strip() if line else ""




    def _condense_sentences_to_summary(self, section_sentences: List[str]) -> str:
        """Take per-section sentences and produce one short paragraph: no repetition of terms, consistent and concise."""
        if not section_sentences:
            return ""
        client = OpenAI()
        sentences_text = "\n".join(f"- {s}" for s in section_sentences)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are given one sentence per section of an academic paper. Write ONE short paragraph "
                        "that summarizes the document. Do NOT repeat the same ideas,"
                        "merge and rephrase so each point is said once. The paragraph should read as one consistent "
                        "summary, concise and with no filler. Include key terms (models, datasets, methods, results). "
                        "Reply with only the paragraph, nothing else."
                    ),
                },
                {"role": "user", "content": f"Section sentences:\n{sentences_text}"},
            ],
            temperature=0,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return summary


    def _is_proper_chunk_text(self, content: str) -> bool:
        """True only if content looks like real prose: sentence structure, word shape, no broken PDF or number runs."""
        s = content.strip()
        words = [w for w in s.split() if w]
        if not s or len(s) < 35 or len(words) < 5:
            return False
        n, nw = len(s), len(words)
        digits = sum(1 for c in s if c.isdigit()) / max(n, 1)
        single = sum(1 for w in words if len(w) == 1) / max(nw, 1)
        alpha = sum(1 for w in words if any(c.isalpha() for c in w)) / nw
        unique = len(set(w.lower() for w in words)) / max(nw, 1)
        avg_len = sum(len(w) for w in words) / max(nw, 1)
        has_end = any(c in s for c in ".!?")
        return (
            digits <= 0.4 and single <= 0.28 and alpha >= 0.5 and unique >= 0.3
            and (has_end or avg_len >= 3.2)
            and not (s.isupper() and n < 80)
        )

    def _chunk_section(self, text: str) -> List[str]:
        """Chunk text using LlamaIndex SentenceSplitter (chunk_size=100, overlap=20)."""
        text = text.strip()
        if not text:
            return []
        doc = LlamaIxDocument(text=text)
        nodes = self.sentence_splitter.get_nodes_from_documents([doc], show_progress=False)
        return [node.get_content() for node in nodes if node.get_content().strip()]





    def process_document(self, doc_path: str) -> Tuple[Document, List[RAGChunk]]:
        """
        Extract title from PDF. Collect text per section (and per subsection when present).
        Chunk each section/subsection with SentenceSplitter (size=100, overlap=20).
        Returns (Document, List[RAGChunk]) with section/subsection set; embedding and knowledge_role filled by embed_chunks.
        """
        arxiv_id = self._arxiv_id_from_path(doc_path)
        doc_id = arxiv_id
        title, abstract = self._extract_title_and_abstract_from_pdf(doc_path)
        # keywords = self._generate_keywords_with_gpt(abstract) if abstract else []


        reader = PdfReader(doc_path)
        chunks_out: List[RAGChunk] = []
        current_section: Optional[str] = "Abstract"
        current_subsection: Optional[str] = None
        section_content: List[str] = []
        section_header = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")

        section_sentences: List[str] = []


        def flush():
            if not section_content:
                return
            combined = " ".join(section_content).strip()
            if not combined:
                return

            # One sentence per section (condense stage later removes repetition)
            usage_line = self._extract_section_usage_line(combined)
            if usage_line:
                section_sentences.append(usage_line)

            # Split the section into chunks (embeddings filled later by embed_chunks in one batch)
            for content in self._chunk_section(combined):
                if content and self._is_proper_chunk_text(content):
                    chunks_out.append(
                        RAGChunk(
                            doc_id=doc_id,
                            content=content,
                            section=current_section,
                            subsection=current_subsection,
                        )
                    )
            section_content.clear()


        # First page: start from after the "Abstract" line (skip title, authors, and "Abstract" header)
        first_page_text = (reader.pages[0].extract_text() or "").strip()
        first_page_lines = first_page_text.split("\n")
        abstract_start = None
        for i, ln in enumerate(first_page_lines):
            if re.match(r"^\s*abstract\s*$", ln.strip(), re.IGNORECASE):
                abstract_start = i + 1
                break
        if abstract_start is not None:
            first_page_from_abstract = "\n".join(first_page_lines[abstract_start:]).strip()
        else:
            first_page_from_abstract = first_page_text

        # Full document: abstract body + rest of pages (section content can span many pages)
        rest_pages = "\n\n".join(
            (page.extract_text() or "").strip() for page in list(reader.pages)[1:]
        )
        full_text = first_page_from_abstract + ("\n\n" + rest_pages if rest_pages.strip() else "")
        blocks = [b.strip() for b in full_text.split("\n\n") if b.strip()]



        def _is_references_or_bibliography(title: str) -> bool:
            t = title.lower()
            return "references" in t or "bibliography" in t



        def _is_standalone_references_header(line: str) -> bool:
            """References section header only: 'References' / 'Bibliography' or '6 References', not titles like 'Learning to Retrieve References'."""
            s = line.strip()
            if not s or len(s.split()) > 10:
                return False
            t = s.lower()
            if t in ("references", "bibliography"):
                return True
            if re.match(r"^\d+(\.\d+)*\s+(references|bibliography)(\s|$)", t):
                return True
            if len(s.split()) <= 4 and (t.startswith("references ") or t.startswith("bibliography ")):
                return True
            return False

        def _is_appendix_header(line: str) -> bool:
            """True if line looks like a page/section title containing 'Appendix' or 'Appendices' — resume chunking after references."""
            s = line.strip()
            if not s:
                return False
            t = s.lower()
            if "appendix" not in t and "appendices" not in t:
                return False
            # Title-like: short enough to be a heading, not body text
            words = s.split()
            return len(words) <= 15 and len(s) <= 120

        in_references = False
        for block in blocks:
            lines_in_block = block.split("\n")
            current_part = []
            for line in lines_in_block:
                line_stripped = line.strip()
                if in_references:
                    if _is_appendix_header(line_stripped):
                        in_references = False
                        if current_part:
                            section_content.append("\n".join(current_part))
                        flush()
                        match = section_header.match(line_stripped)
                        if match and len(line_stripped.split()) <= 15:
                            if "." in match.groups()[0]:
                                current_subsection = line_stripped
                            else:
                                current_section = line_stripped
                                current_subsection = None
                        else:
                            current_section = line_stripped
                            current_subsection = None
                        current_part = []
                    continue
                if _is_standalone_references_header(line_stripped):
                    if current_part:
                        section_content.append("\n".join(current_part))
                    flush()
                    in_references = True
                    continue
                match = section_header.match(line_stripped)
                if match and len(line_stripped.split()) <= 15:
                    section_title = match.group(2).strip()
                    if current_part:
                        section_content.append("\n".join(current_part))
                    flush()
                    if _is_references_or_bibliography(section_title):
                        in_references = True
                        continue
                    if "." in match.groups()[0]:
                        current_subsection = line_stripped
                    else:
                        current_section = line_stripped
                        current_subsection = None
                    current_part = []
                else:
                    current_part.append(line)
            if not in_references and current_part:
                section_content.append("\n".join(current_part))
        if not in_references:
            flush()

        # Condense per-section sentences into one short paragraph (no repetition, consistent)
        document_summary = self._condense_sentences_to_summary(section_sentences)
        document_summary_embedding = self._embed_text(document_summary)
        document = Document(
            doc_id=doc_id,
            arxiv_id=arxiv_id,
            title=title,
            document_summary=document_summary,
            document_summary_embedding=document_summary_embedding,
        )
        

        return document, chunks_out




    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text with OpenAI text-embedding-3-small. Returns None for empty text."""
        if not text or not text.strip():
            return None
        client = OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text.strip()],
        )
        if not resp.data:
            return None
        return resp.data[0].embedding




    def embed_chunks(self, document: Document, chunks: List[RAGChunk]) -> List[RAGChunk]:
        """Embed all chunks in one batch (OpenAI text-embedding-3-small) and fill chunk.embedding."""
        if not chunks:
            return []
        client = OpenAI()
        texts = [c.content for c in chunks]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = [e.embedding for e in resp.data]
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks




    @staticmethod
    def _sanitize_text_for_db(s: Optional[str]) -> Optional[str]:
        """Remove null bytes and other control chars PostgreSQL text columns reject."""
        if s is None:
            return None
        return "".join(c for c in s if c != "\x00" and (ord(c) >= 32 or c in "\n\r\t"))

    def store_document(self, document: Document) -> None:
        """Insert or upsert the document into Supabase table 'documents'."""
        row = {
            "doc_id": document.doc_id,
            "arxiv_id": document.arxiv_id,
            "title": self._sanitize_text_for_db(document.title) or "",
            "document_summary": self._sanitize_text_for_db(document.document_summary) or "",
            "document_summary_embedding": document.document_summary_embedding,
        }
        self.supabase_client.table("documents").upsert(
            row,
            on_conflict="doc_id",
        ).execute()




    def store_chunks(self, chunks: List[RAGChunk]) -> None:
        """Delete existing chunks for this doc, then insert the new chunks."""
        if not chunks:
            return
        doc_id = chunks[0].doc_id
        self.supabase_client.table("chunks").delete().eq("doc_id", doc_id).execute()
        rows = [
            {
                "doc_id": chunk.doc_id,
                "content": self._sanitize_text_for_db(chunk.content) or "",
                "embedding": chunk.embedding,
                "section": self._sanitize_text_for_db(chunk.section),
                "subsection": self._sanitize_text_for_db(chunk.subsection),
            }
            for chunk in chunks
        ]
        self.supabase_client.table("chunks").insert(rows).execute()




    def process_and_store_docs(self):
        for doc_path in self.documents:
            print(f"Processing {doc_path}")
            document, chunks = self.process_document(doc_path)
            chunks = self.embed_chunks(document, chunks)
            
            self.store_document(document)
            self.store_chunks(chunks)







