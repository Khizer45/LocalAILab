"""Request bodies (Pydantic). Responses are plain dicts built by serializers in
`app.schemas.serializers`, so the API shape is defined in one place."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=1000)
    num_ctx: int | None = Field(None, ge=256, le=262144)
    num_predict: int | None = Field(None, ge=1, le=32768)
    seed: int | None = None
    repeat_penalty: float | None = Field(None, ge=0.5, le=2.5)


class ConversationCreate(BaseModel):
    title: str | None = Field(None, max_length=200)
    model: str | None = Field(None, max_length=200)
    system_prompt: str | None = Field(None, max_length=8000)
    options: GenerationOptions | None = None
    use_documents: bool = False


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, max_length=200)
    model: str | None = Field(None, max_length=200)
    system_prompt: str | None = Field(None, max_length=8000)
    options: GenerationOptions | None = None
    use_documents: bool | None = None


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    model: str | None = Field(None, max_length=200)
    options: GenerationOptions | None = None
    use_documents: bool | None = None
    document_ids: list[int] | None = None
    think: bool | None = None


class AskDocuments(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(None, max_length=200)
    document_ids: list[int] | None = None
    options: GenerationOptions | None = None


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=256)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
