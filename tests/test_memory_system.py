"""
Unit and integration tests for Mem0-inspired Memory System in IT-Law-Chatbot.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.memory import (
    ShortTermMemory,
    LongTermMemory,
    MemoryExtractor,
    get_short_term_memory,
    get_long_term_memory,
)
from app.api.routes.memory import memory_router

# Lightweight test app mounting memory_router
test_app = FastAPI()
test_app.include_router(memory_router)
client = TestClient(test_app)


def test_short_term_memory_session_tracking():
    """Test session state lifecycle and citation extraction in short-term memory."""
    st = ShortTermMemory()
    conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"

    state = st.get_or_create_state(conv_id, user_id="user_test_1")
    assert state.conversation_id == conv_id
    assert state.turn_count == 0

    # Simulate bot response with citations
    dummy_answer = (
        "Căn cứ vào Điều 84 Nghị định 15/2020/NĐ-CP, hành vi thu thập thông tin cá nhân trái phép "
        "bị xử phạt tiền từ 10.000.000 đến 20.000.000 đồng."
    )
    dummy_sources = [{
        "doc_title": "Nghị định 15/2020/NĐ-CP",
        "article": "Điều 84",
        "dieu_so": "84",
        "score": 0.95
    }]

    st.extract_citations_and_update(
        conversation_id=conv_id,
        query="Công ty tôi thu thập dữ liệu người dùng thì có bị phạt không?",
        answer=dummy_answer,
        sources=dummy_sources,
        user_id="user_test_1",
    )

    state = st.get_state(conv_id)
    assert state is not None
    assert state.turn_count == 1
    assert "Nghị định 15/2020/NĐ-CP" in state.focused_laws
    assert "Điều 84" in state.focused_articles
    assert state.user_role == "tổ chức / doanh nghiệp"


def test_short_term_fast_path_enrichment():
    """Test Fast-Path query detection and context enrichment without calling LLM rewrite."""
    st = ShortTermMemory()
    conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"

    # Setup state with focused law & article
    st.update_state(
        conversation_id=conv_id,
        focused_laws=["Nghị định 15/2020/NĐ-CP"],
        focused_articles=["Điều 84"],
        user_role="cá nhân",
        topic="thu thập dữ liệu cá nhân",
        user_id="user_test_2",
    )

    # Test follow-up query detection
    follow_up_1 = "mức phạt đối với cá nhân thế nào?"
    follow_up_2 = "vậy ai có thẩm quyền xử phạt?"
    regular_query = "Quy định về bảo hộ phần mềm theo Luật Sở hữu trí tuệ 2005"

    assert st.is_follow_up_query(follow_up_1, conv_id) is True
    assert st.is_follow_up_query(follow_up_2, conv_id) is True
    assert st.is_follow_up_query(regular_query, conv_id) is False

    # Test Fast-Path query enrichment
    enriched_1 = st.build_fast_path_query(follow_up_1, conv_id)
    assert "Điều 84" in enriched_1
    assert "Nghị định 15/2020/NĐ-CP" in enriched_1

    enriched_2 = st.build_fast_path_query(follow_up_2, conv_id)
    assert "Điều 84" in enriched_2
    assert "Nghị định 15/2020/NĐ-CP" in enriched_2


def test_long_term_memory_deduplication_and_crud():
    """Test LongTermMemory addition, semantic deduplication, search, and deletion in Qdrant."""
    lt = get_long_term_memory()
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    conv_id = f"conv_{uuid.uuid4().hex[:8]}"

    # 1. Add first fact
    fact_1 = "Người dùng là doanh nghiệp phát triển ứng dụng di động có thu thập dữ liệu cá nhân."
    res1 = lt.add_memory(
        fact=fact_1,
        user_id=user_id,
        conversation_id=conv_id,
        memory_type="user_profile",
        entities=["ứng dụng di động", "dữ liệu cá nhân"],
    )
    assert res1 is not None
    assert res1["fact"] == fact_1
    first_id = res1["id"]

    # 2. Add very similar fact (semantic deduplication check)
    fact_2 = "Người dùng là công ty phát triển ứng dụng di động có thu thập dữ liệu cá nhân người dùng."
    res2 = lt.add_memory(
        fact=fact_2,
        user_id=user_id,
        conversation_id=conv_id,
        memory_type="user_profile",
        entities=["ứng dụng di động", "dữ liệu cá nhân"],
    )
    assert res2 is not None
    assert res2["id"] == first_id  # Should update existing point, not create duplicate!
    assert res2["update_count"] >= 2

    # 3. Search memories
    memories = lt.search_memories(
        query="công ty phát triển ứng dụng di động có thu thập dữ liệu cá nhân",
        user_id=user_id,
        limit=3,
        min_score=0.25,
    )
    assert len(memories) >= 1
    assert memories[0]["id"] == first_id

    # 4. Format for prompt
    prompt_context = LongTermMemory.format_memories_for_prompt(memories)
    assert "[Hồ sơ người dùng]" in prompt_context

    # 5. Get all & delete
    all_mems = lt.get_all_memories(user_id=user_id)
    assert len(all_mems) == 1

    del_res = lt.delete_memory(first_id)
    assert del_res is True

    # 6. Verify clear
    all_mems_after = lt.get_all_memories(user_id=user_id)
    assert len(all_mems_after) == 0


def test_memory_api_endpoints():
    """Test REST API routes for memory management."""
    user_id = f"api_user_{uuid.uuid4().hex[:8]}"

    # 1. Add memory via API
    add_payload = {
        "fact": "Người dùng là đơn vị cung cấp dịch vụ sàn thương mại điện tử xuyên biên giới.",
        "user_id": user_id,
        "memory_type": "user_profile",
        "entities": ["sàn TMĐT", "xuyên biên giới"],
    }
    response = client.post("/api/memory", json=add_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    memory_id = data["data"]["id"]

    # 2. Get user memories via API
    response = client.get(f"/api/memory/user/{user_id}")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert len(res_data["data"]) >= 1

    # 3. Get session memory
    dummy_conv_id = f"conv_api_{uuid.uuid4().hex[:8]}"
    st = get_short_term_memory()
    st.update_state(
        conversation_id=dummy_conv_id,
        focused_laws=["Nghị định 52/2013/NĐ-CP"],
        focused_articles=["Điều 36"],
        user_role="doanh nghiệp sàn TMĐT",
        user_id=user_id,
    )

    response = client.get(f"/api/memory/session/{dummy_conv_id}")
    assert response.status_code == 200
    session_data = response.json()
    assert session_data["success"] is True
    assert "Nghị định 52/2013/NĐ-CP" in session_data["data"]["focused_laws"]

    # 4. Clear user memories
    response = client.delete(f"/api/memory/user/{user_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True


if __name__ == "__main__":
    print("Running memory system tests...")
    test_short_term_memory_session_tracking()
    print("[PASS] Short-term memory session tracking passed.")
    test_short_term_fast_path_enrichment()
    print("[PASS] Short-term Fast-Path enrichment passed.")
    test_long_term_memory_deduplication_and_crud()
    print("[PASS] Long-term memory deduplication and CRUD passed.")
    test_memory_api_endpoints()
    print("[PASS] Memory REST API endpoints passed.")
    print("\nALL MEMORY SYSTEM TESTS PASSED SUCCESSFULLY! [OK]")
