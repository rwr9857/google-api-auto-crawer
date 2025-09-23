import torch
import requests
from langchain_huggingface import HuggingFaceEmbeddings

BASE_URL = "http://localhost:8000/api/v2"
COLLECTION_NAME = "google_api_docs"

resp = requests.request("GET", f"{BASE_URL}/healthcheck")
print(f"✅ {resp.status_code} 서버 정상 실행")
print("=" * 50)

# ==============================
# 사용자 정보 확인 & tenant/database 설정
# ==============================
resp = requests.get(f"{BASE_URL}/auth/identity")
identity = resp.json()


TENANT_NAME = identity["tenant"]
# 첫 번째 데이터베이스 사용
DATABASE_NAME = identity["databases"][0]

print(f"✅ 연결 성공")
print(f"   Tenant: {TENANT_NAME}")
print(f"   Database: {DATABASE_NAME}")
print("=" * 50)

# ==============================
# 기존 컬렉션 목록 조회
# ==============================
resp = requests.get(
    f"{BASE_URL}/tenants/{TENANT_NAME}/databases/{DATABASE_NAME}/collections"
)
collections = resp.json()

# ==============================
# 원하는 컬렉션이 있는지 확인하고 없으면 생성
# ==============================
collection = next((c for c in collections if c["name"] == COLLECTION_NAME), None)

if collection:
    print(f"✅ 기존 컬렉션 발견")
    collection_id = collection["id"]
    print(f"   ID: {collection_id}")
    print(f"   Dimension: {collection.get('dimension', 'N/A')}")
else:
    print(f"📝 컬렉션 '{COLLECTION_NAME}'이 없어서 생성합니다...")

    # 컬렉션 생성
    collection_data = {
        "name": COLLECTION_NAME,
        "metadata": {"description": "Google API documentation collection"},
    }

    resp = requests.post(
        f"{BASE_URL}/tenants/{TENANT_NAME}/databases/{DATABASE_NAME}/collections",
        json=collection_data,
    )
    resp.raise_for_status()

    collection = resp.json()
    collection_id = collection["id"]
    print(f"✅ 컬렉션 생성 완료")
    print(f"   ID: {collection_id}")

# 최종 정보 출력
print(f"\n✅ 사용 가능한 컬렉션 정보:")
print(f"   이름: {COLLECTION_NAME}")
print(f"   ID: {collection_id}")
print(
    f"   경로: /tenants/{TENANT_NAME}/databases/{DATABASE_NAME}/collections/{collection_id}"
)
print("=" * 50)


# ===============================
# HTTP API 쿼리 조회
# ===============================
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
query_vec = embeddings.embed_query("map")

collection_url = f"{BASE_URL}/tenants/{TENANT_NAME}/databases/{DATABASE_NAME}/collections/{collection_id}"

res = requests.post(
    f"{collection_url}/query",
    json={
        "query_embeddings": [query_vec],
        "n_results": 1,
        "include": ["documents", "metadatas", "distances"],
    },
)
result = res.json()

# ===============================
# 결과 flatten
# ===============================
flatten_docs = []
flatten_metas = []
flatten_distances = []

# result는 2중 리스트 구조: [쿼리 수][검색 결과 수]
for doc_list, meta_list, dist_list in zip(
    result.get("documents", []),
    result.get("metadatas", []),
    result.get("distances", []),
):
    for doc, meta, dist in zip(doc_list, meta_list, dist_list):
        flatten_docs.append(doc)
        flatten_metas.append(meta)
        flatten_distances.append(dist)

# ===============================
# 출력
# ===============================
for i, (doc, meta, dist) in enumerate(
    zip(flatten_docs, flatten_metas, flatten_distances), 1
):
    print(
        f"\n[{i}] 거리: {dist:.4f}, 파일: {meta.get('source_file')}, 청크 ID: {meta.get('chunk_id')}"
    )
    print(f"태그: {meta.get('tags')}, 원본 URL: {meta.get('source')}")
    print("내용:\n", doc)
    print("-" * 60)


# ===============================
# 텍스트 기반 검색
# ===============================
payload = {
    # "where": "map",
    # "where_document": '{"$contains": "map"}',
    # "ids": ["string"],
    "include": ["metadatas", "documents", "distances"],
    "limit": 10,
    "offset": 10,
}

res = requests.post(f"{collection_url}/get", json=payload)
res.raise_for_status()

result = res.json()
print(f"\n\n✅ 텍스트 기반 검색 결과 (총 {result.get('count', 0)}개):")

for i, query in enumerate(result.get("results", []), 1):
    print(f"=== Query #{i} ===")
    for j, doc in enumerate(query.get("documents", []), 1):
        metadata = query.get("metadatas", [])[j - 1] if query.get("metadatas") else {}
        print(f"[{j}] 파일: {metadata.get('source_file')}")
        print("내용:\n", doc)
        print("-" * 60)
