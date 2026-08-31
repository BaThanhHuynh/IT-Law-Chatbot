"""
Long-Term Vector Memory (Mem0-inspired) backed by Qdrant.
Stores episodic facts, user profiles, and legal interests with semantic search & deduplication.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from app.core.config import Config
from app.core.logger import logger
from app.services.rag.embeddings import get_embedding
from app.services.rag.retriever import get_qdrant_client
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct, Filter,
    FieldCondition, MatchValue, PayloadSchemaType, PointIdsList, FilterSelector
)


class LongTermMemory:
    """Manages persistent semantic facts and user profiles in Qdrant vector database."""

    def __init__(self):
        self._initialized = False

    def _init_collection(self):
        if self._initialized:
            return
        client = get_qdrant_client()
        try:
            collections = [c.name for c in client.get_collections().collections]
            if Config.QDRANT_MEMORY_COLLECTION not in collections:
                client.create_collection(
                    collection_name=Config.QDRANT_MEMORY_COLLECTION,
                    vectors_config=VectorParams(size=Config.EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info(f"[LongTermMemory] Created Qdrant collection '{Config.QDRANT_MEMORY_COLLECTION}'.")

            # Ensure payload indexes on user_id and memory_type
            try:
                client.create_payload_index(
                    collection_name=Config.QDRANT_MEMORY_COLLECTION,
                    field_name="user_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

            try:
                client.create_payload_index(
                    collection_name=Config.QDRANT_MEMORY_COLLECTION,
                    field_name="memory_type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

            self._initialized = True
        except Exception as e:
            logger.error(f"[LongTermMemory] Failed to initialize memory collection: {e}")

    def add_memory(
        self,
        fact: str,
        user_id: str = "default_user",
        conversation_id: Optional[str] = None,
        memory_type: str = "fact",
        entities: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add or update a fact in long-term memory.
        Performs semantic deduplication (Mem0 style): if an existing memory has cosine similarity >= 0.85,
        updates the existing record rather than creating duplicate points.
        """
        fact = fact.strip()
        if not fact or len(fact) < 5:
            return None

        self._init_collection()
        client = get_qdrant_client()
        fact_embedding = get_embedding(fact)
        now = datetime.now().isoformat()

        # Step 1: Check for highly similar existing facts (deduplication / merge)
        try:
            response = client.query_points(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                query=fact_embedding.tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=1,
                score_threshold=0.85,
                with_payload=True,
            )
            similar = response.points

            if similar:
                existing_point = similar[0]
                existing_id = existing_point.id
                existing_payload = existing_point.payload or {}
                
                # Update existing memory with latest timestamp and merged facts
                old_entities = existing_payload.get("entities", [])
                merged_entities = list(set(old_entities + (entities or [])))
                
                updated_payload = {
                    **existing_payload,
                    "fact": fact,
                    "memory_type": memory_type or existing_payload.get("memory_type", "fact"),
                    "entities": merged_entities,
                    "updated_at": now,
                    "update_count": existing_payload.get("update_count", 1) + 1,
                }

                client.upsert(
                    collection_name=Config.QDRANT_MEMORY_COLLECTION,
                    points=[
                        PointStruct(
                            id=existing_id,
                            vector=fact_embedding.tolist(),
                            payload=updated_payload,
                        )
                    ],
                )
                logger.info(f"[LongTermMemory] Merged/Updated existing memory: '{fact}' (score={existing_point.score:.3f})")
                return updated_payload

        except Exception as e:
            logger.warning(f"[LongTermMemory] Dedup check failed, inserting as new: {e}")

        # Step 2: Insert as new point
        memory_id = str(uuid.uuid4())
        payload = {
            "id": memory_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "fact": fact,
            "memory_type": memory_type,
            "entities": entities or [],
            "created_at": now,
            "updated_at": now,
            "update_count": 1,
        }

        try:
            client.upsert(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=fact_embedding.tolist(),
                        payload=payload,
                    )
                ],
            )
            logger.info(f"[LongTermMemory] Saved new memory: '{fact}' for user '{user_id}'")
            return payload
        except Exception as e:
            logger.error(f"[LongTermMemory] Failed to save memory: {e}")
            return None

    def search_memories(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 3,
        min_score: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant long-term memories for a user query.
        """
        self._init_collection()
        client = get_qdrant_client()
        query_embedding = get_embedding(query)

        try:
            response = client.query_points(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                query=query_embedding.tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            )
            results = response.points

            memories = []
            for hit in results:
                payload = hit.payload or {}
                memories.append({
                    "id": hit.id,
                    "fact": payload.get("fact", ""),
                    "memory_type": payload.get("memory_type", "fact"),
                    "entities": payload.get("entities", []),
                    "score": round(hit.score, 3),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "update_count": payload.get("update_count", 1),
                })
            
            if memories:
                logger.info(f"[LongTermMemory] Found {len(memories)} memories for query (Top: '{memories[0]['fact']}' score={memories[0]['score']})")
            return memories
        except Exception as e:
            logger.error(f"[LongTermMemory] Search memories failed: {e}")
            return []

    def get_all_memories(self, user_id: str = "default_user", limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all memories for a specific user."""
        self._init_collection()
        client = get_qdrant_client()
        try:
            records, _ = client.scroll(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=limit,
                with_payload=True,
            )
            memories = [r.payload for r in records if r.payload]
            memories.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return memories
        except Exception as e:
            logger.error(f"[LongTermMemory] get_all_memories failed: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item by ID."""
        self._init_collection()
        client = get_qdrant_client()
        try:
            client.delete(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                points_selector=PointIdsList(points=[memory_id]),
            )
            logger.info(f"[LongTermMemory] Deleted memory ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"[LongTermMemory] delete_memory failed: {e}")
            return False

    def clear_user_memories(self, user_id: str = "default_user") -> bool:
        """Delete all memories for a user."""
        self._init_collection()
        client = get_qdrant_client()
        try:
            client.delete(
                collection_name=Config.QDRANT_MEMORY_COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                    )
                ),
            )
            logger.info(f"[LongTermMemory] Cleared all memories for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"[LongTermMemory] clear_user_memories failed: {e}")
            return False

    @staticmethod
    def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
        """Format a list of memory records into a prompt context string."""
        if not memories:
            return ""
        lines = []
        for m in memories:
            fact_text = m.get("fact", "")
            m_type = m.get("memory_type", "fact")
            if fact_text:
                if m_type == "user_profile":
                    lines.append(f"- [Hồ sơ người dùng] {fact_text}")
                elif m_type == "legal_context":
                    lines.append(f"- [Ngữ cảnh pháp lý] {fact_text}")
                else:
                    lines.append(f"- {fact_text}")
        return "\n".join(lines)
