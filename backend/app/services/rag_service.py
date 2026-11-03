"""Grounded answer preparation (retrieval + safe prompt construction)."""

from __future__ import annotations

from app.core.errors import OfflineModeViolation
from app.services.document_service import DocumentService
from app.services.model_registry import ResolvedModel
from app.services.prompt_safety import RAG_SYSTEM_POLICY, build_rag_user_message
from app.services.settings_service import SettingsService

NOT_FOUND_ANSWER = "I could not find this in the indexed documents."


def public_source(s: dict) -> dict:
    """Client-facing view of a retrieved chunk."""
    return {k: s[k] for k in ("n", "document_id", "filename", "page", "chunk_index", "score", "flagged")} | {
        "snippet": s["text"][:600]
    }


class RagService:
    def __init__(self, documents: DocumentService, settings_service: SettingsService):
        self.documents = documents
        self.settings_service = settings_service

    def prepare(self, owner_id: int, question: str, model: ResolvedModel, document_ids: list[int] | None = None,
                user_system_prompt: str = "") -> dict:
        rs = self.settings_service.get()
        if model.runs_in_cloud and not rs.allow_documents_with_cloud:
            raise OfflineModeViolation(
                "Document Q&A with a cloud model is disabled, because document excerpts would leave this computer.",
                code="documents_cloud_blocked",
                hint="Use a local model, or enable 'Allow documents with cloud models' in Settings.",
            )
        r = self.documents.retrieve(owner_id, question, top_k=rs.retrieval_top_k, document_ids=document_ids)
        sources = r["sources"]
        warnings: list[str] = []
        if r["embedder_note"]:
            warnings.append(r["embedder_note"])
        if r["stale_documents"]:
            warnings.append(
                f"{len(r['stale_documents'])} document(s) were indexed with a different embedding model and were "
                "not searched. Use 'Re-index' on the Documents page."
            )
        for s in sources:
            if s["flagged"]:
                warnings.append(
                    f"Source S{s['n']} ({s['filename']}) contains instruction-like text (possible prompt injection). "
                    "It was given to the model as data only."
                )
        system = RAG_SYSTEM_POLICY
        if user_system_prompt.strip():
            system += "\n\nStyle preferences from the user (lower priority than the rules above):\n" + user_system_prompt.strip()
        return {
            "grounded": bool(sources),
            "sources": sources,
            "public_sources": [public_source(s) for s in sources],
            "warnings": warnings,
            "system": system,
            "user_message": build_rag_user_message(question, sources) if sources else None,
            "retrieval": {k: r[k] for k in ("embedder", "min_score", "searched_documents")},
        }
