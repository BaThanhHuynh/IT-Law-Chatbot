"""API routes for the IT Law Chatbot."""
import json
from flask import Blueprint, request, jsonify
from chatbot.engine import (
    generate_response,
    create_conversation,
    get_conversation_history,
    get_all_conversations,
)
from graphrag.knowledge_graph import get_knowledge_graph

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    """Send a message and get AI response."""
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400

    query = data["message"]
    conversation_id = data.get("conversation_id")

    try:
        result = generate_response(query, conversation_id)
        return jsonify({
            "success": True,
            "data": {
                "conversation_id": result["conversation_id"],
                "answer": result["answer"],
                "sources": result["sources"],
                "graph_data": result["graph_data"],
            }
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@chat_bp.route("/api/conversations", methods=["GET"])
def list_conversations():
    """Get all conversations."""
    try:
        conversations = get_all_conversations()
        # Convert datetime objects to strings
        for conv in conversations:
            for key in ["created_at", "updated_at"]:
                if conv.get(key):
                    conv[key] = conv[key].isoformat()
        return jsonify({"success": True, "data": conversations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/conversations", methods=["POST"])
def new_conversation():
    """Create a new conversation."""
    try:
        conv_id = create_conversation()
        return jsonify({"success": True, "data": {"id": conv_id}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Get messages for a conversation."""
    try:
        messages = get_conversation_history(conversation_id)
        for msg in messages:
            if msg.get("created_at"):
                msg["created_at"] = msg["created_at"].isoformat()
            if msg.get("sources") and isinstance(msg["sources"], str):
                msg["sources"] = json.loads(msg["sources"])
        return jsonify({"success": True, "data": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/api/knowledge-graph", methods=["GET"])
def get_kg_data():
    """Get knowledge graph data for visualization."""
    try:
        kg = get_knowledge_graph()
        kg.ensure_loaded()

        # Optional: filter by entity_ids
        entity_ids = request.args.get("entity_ids")
        if entity_ids:
            entity_ids = entity_ids.split(",")

        graph_data = kg.get_graph_data_for_visualization(
            entity_ids=entity_ids,
            depth=int(request.args.get("depth", 1))
        )
        return jsonify({"success": True, "data": graph_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
