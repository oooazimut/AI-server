from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re

from pydantic import BaseModel, Field

from .embeddings import EmbeddingProvider, create_embedding_provider
from .knowledge import MarkdownKnowledgeBase
from .models import AgentManifest


class RetrievalChunk(BaseModel):
    id: str
    agent_id: str
    topic: str
    title: str
    section: str
    text: str
    path: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk: RetrievalChunk
    score: float
    keyword_score: float
    vector_score: float
    embedding_provider: str = ""
    matched_terms: list[str] = Field(default_factory=list)


class HybridKnowledgeRetriever:
    def __init__(
        self,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        keyword_weight: float = 0.65,
        vector_weight: float = 0.35,
    ) -> None:
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight

    def search(
        self,
        manifest: AgentManifest,
        query: str,
        *,
        limit: int = 5,
        topic: str | None = None,
    ) -> list[RetrievalHit]:
        query = query.strip()
        if not query:
            return []

        chunks = self._chunks(manifest, topic=topic)
        if not chunks:
            return []

        query_terms = _tokenize(query)
        idf = _inverse_document_frequency(chunks)
        query_vector = self.embedding_provider.embed(query)

        raw_hits: list[RetrievalHit] = []
        max_keyword_score = 0.0
        for chunk in chunks:
            keyword_score, matched_terms = _keyword_score(chunk, query_terms, idf)
            max_keyword_score = max(max_keyword_score, keyword_score)
            vector_score = _cosine(query_vector, self.embedding_provider.embed(_chunk_vector_text(chunk)))
            raw_hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=0.0,
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                    embedding_provider=self.embedding_provider.name,
                    matched_terms=matched_terms,
                )
            )

        hits: list[RetrievalHit] = []
        for hit in raw_hits:
            normalized_keyword = hit.keyword_score / max_keyword_score if max_keyword_score else 0.0
            score = self.keyword_weight * normalized_keyword + self.vector_weight * hit.vector_score
            if score <= 0:
                continue
            hit.score = round(score, 6)
            hit.keyword_score = round(normalized_keyword, 6)
            hit.vector_score = round(hit.vector_score, 6)
            hits.append(hit)

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, min(limit, 20))]

    def _chunks(self, manifest: AgentManifest, *, topic: str | None) -> list[RetrievalChunk]:
        selected_topic = topic.strip().lower() if topic else None
        chunks: list[RetrievalChunk] = []
        for item in self.knowledge_base.list_topics(manifest):
            if selected_topic and item.name != selected_topic:
                continue
            path = Path(item.path)
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            for index, section in enumerate(_split_markdown_sections(content), start=1):
                chunks.append(
                    RetrievalChunk(
                        id=f"{manifest.id}:{item.name}:{section.slug or index}",
                        agent_id=manifest.id,
                        topic=item.name,
                        title=item.title,
                        section=section.title,
                        text=section.text,
                        path=str(path),
                        metadata={"section_slug": section.slug},
                    )
                )
        return chunks


class _MarkdownSection(BaseModel):
    slug: str
    title: str
    text: str


def _split_markdown_sections(content: str) -> list[_MarkdownSection]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", content))
    if not matches:
        return [_MarkdownSection(slug="document", title="Документ", text=content)]

    sections: list[_MarkdownSection] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            sections.append(_MarkdownSection(slug=_slugify(title), title=title, text=text))
    return sections


def _keyword_score(chunk: RetrievalChunk, query_terms: list[str], idf: dict[str, float]) -> tuple[float, list[str]]:
    if not query_terms:
        return 0.0, []

    title_terms = Counter(_tokenize(chunk.title + " " + chunk.section))
    body_terms = Counter(_tokenize(chunk.text))
    score = 0.0
    matched: list[str] = []
    for term in query_terms:
        title_count = title_terms.get(term, 0)
        body_count = body_terms.get(term, 0)
        if not title_count and not body_count:
            continue
        matched.append(term)
        score += idf.get(term, 1.0) * (3.0 * title_count + body_count)
    return score, matched


def _inverse_document_frequency(chunks: list[RetrievalChunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(_tokenize(_chunk_vector_text(chunk))))

    total = len(chunks)
    return {
        term: math.log((total + 1) / (count + 0.5)) + 1.0
        for term, count in document_frequency.items()
    }


def _chunk_vector_text(chunk: RetrievalChunk) -> str:
    return f"{chunk.title}\n{chunk.section}\n{chunk.text}"


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[0-9a-zа-яё_\.]{2,}", _normalize(value), flags=re.IGNORECASE)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _slugify(value: str) -> str:
    normalized = _normalize(value)
    return re.sub(r"[^0-9a-zа-яё]+", "-", normalized, flags=re.IGNORECASE).strip("-") or "section"
