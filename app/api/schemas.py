from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., description="Câu hỏi của người dùng")
    conversation_id: Optional[str] = Field(None, description="ID cuộc hội thoại nếu đã có")
    user_id: Optional[str] = Field("default_user", description="ID người dùng để cá nhân hóa bộ nhớ")

class Source(BaseModel):
    article: str
    content: str
    score: float
    doc_title: str

class ChatResponseData(BaseModel):
    conversation_id: str
    answer: str
    sources: List[Source]
    graph_data: Dict[str, Any]

class ChatResponse(BaseModel):
    success: bool
    data: ChatResponseData

class ConversationItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ConversationListResponse(BaseModel):
    success: bool
    data: List[ConversationItem]

class NewConversationResponse(BaseModel):
    success: bool
    data: Dict[str, str]

class HistoryResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]

class KGResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class MemoryItem(BaseModel):
    id: str
    user_id: str
    fact: str
    memory_type: str = "fact"
    entities: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    update_count: int = 1

class MemoryAddRequest(BaseModel):
    fact: str = Field(..., min_length=3, description="Nội dung thông tin / fact cần ghi nhớ")
    user_id: Optional[str] = Field("default_user", description="ID người dùng")
    conversation_id: Optional[str] = Field(None, description="ID cuộc hội thoại")
    memory_type: Optional[str] = Field("user_profile", description="Loại bộ nhớ: user_profile, legal_context, fact")
    entities: Optional[List[str]] = Field(default_factory=list, description="Thực thể liên quan")

class MemoryListResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]

class SessionStateResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
