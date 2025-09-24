import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

HOST = "localhost"
PORT = 8000
COLLECTION_NAME = "google_api_docs"
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DEVICE = "cpu"

# Chroma 서버에 붙기
client = chromadb.HttpClient(host=HOST, port=PORT)

# 컬렉션 가져오기
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME, device=EMBED_DEVICE
)

collection = client.get_or_create_collection(
    COLLECTION_NAME, embedding_function=embedding_fn
)
"""
meta_name = "source_file"
docs_name = "youtube_v3_code_samples.txt"

res = collection.get(where={meta_name: docs_name})
print("삭제 전 문서:", len(res["ids"]))

# 특정 source 메타데이터를 가진 문서 삭제
collection.delete(where={meta_name: docs_name})

res = collection.get(where={meta_name: docs_name})
print("삭제 후 남은 문서:", len(res["ids"]))
"""


res = collection.get(where={
    "$or": [
        {"tags": "google_identity"},
        {"tags": "youtube"},
        {"tags": "gmail"},
        {"tags": "calendar"}
    ]
})

print("삭제 전 문서 개수:", len(res["ids"]))


collection.delete(
    where={
        "$or": [
            {"tags": "google_identity"},
            {"tags": "youtube"},
            {"tags": "gmail"},
            {"tags": "calendar"}
        ]
    }
)

res = collection.get(where={
        "$or": [
            {"tags": "google_identity"},
            {"tags": "youtube"},
            {"tags": "gmail"},
            {"tags": "calendar"}
        ]
    })

print("삭제 후 남은 문서:", len(res["ids"]))