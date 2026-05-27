# -*- coding: utf-8 -*-
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

client = QdrantClient(url="http://qdrant:6333")

print("--- SEARCH FOR DIEU SO 94 ---")
res_94, _ = client.scroll(
    collection_name="it_law_chunks",
    scroll_filter=Filter(
        must=[
            FieldCondition(key="dieu_so", match=MatchValue(value="94"))
        ]
    ),
    limit=5
)
for p in res_94:
    pay = p.payload
    print(f"Ten van ban: {pay.get('ten_van_ban')}")
    print(f"Dieu so: {pay.get('dieu_so')}")
    print(f"Dieu ten: {pay.get('dieu_ten')}")
    print(f"Noi dung: {pay.get('noi_dung_chunk')[:300]}")
    print("-" * 50)

print("\n--- SEARCH FOR DIEU SO 80 ---")
res_80, _ = client.scroll(
    collection_name="it_law_chunks",
    scroll_filter=Filter(
        must=[
            FieldCondition(key="dieu_so", match=MatchValue(value="80"))
        ]
    ),
    limit=5
)
for p in res_80:
    pay = p.payload
    print(f"Ten van ban: {pay.get('ten_van_ban')}")
    print(f"Dieu so: {pay.get('dieu_so')}")
    print(f"Dieu ten: {pay.get('dieu_ten')}")
    print(f"Noi dung: {pay.get('noi_dung_chunk')[:300]}")
    print("-" * 50)
