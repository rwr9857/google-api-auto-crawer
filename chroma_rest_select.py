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

# 쿼리
query_text = "bigquery insert"
threshold = 0.1  # 유사도 임계값
k = 10  # 가져올 최대 개수

# 쿼리 실행
res = collection.query(
    query_texts=[query_text],  # 검색할 쿼리
    n_results=k,  # 가져올 결과 개수
    include=["documents", "metadatas", "distances"],
)


def similarity(dist):
    if 0.0 <= dist <= 2.0:
        return 1.0 - (dist / 2.0)
    return None


docs = res.get("documents")[0]
metas = res.get("metadatas")[0]
dists = res.get("distances")[0]

rows = []
for _, meta, dist in zip(docs, metas, dists):
    sim = similarity(dist)
    rows.append(
        {
            "source": meta.get("source"),
            "score": round(sim, 4),
        }
    )

filtered = [r for r in rows if r["score"] >= threshold]

for r in filtered:
    print(r)
