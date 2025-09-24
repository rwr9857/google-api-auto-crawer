import re
import uuid
import chromadb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

api_data_dir = Path("./GOOGLE_API_DATA")
file_paths = list(api_data_dir.rglob("*.txt"))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
)


def _extract_source_url(content: str) -> str:
    """문서에서 Source URL 추출"""
    pattern = r"(?i)Source\s*URL\s*:\s*(https?://\S+)"
    match = re.search(pattern, content)
    return match.group(1).strip() if match else ""


def _get_api_tag_from_path(path) -> str:
    """파일 경로에서 API 태그 추출"""
    folder = path.parent.name
    return (
        folder.replace("_docs_crawled", "")
        if folder.endswith("_docs_crawled")
        else folder
    )


documents = []

# 파일 단위 진행률
today = datetime.today().strftime("%Y-%m-%d")

for file_path in tqdm(file_paths, desc="파일 처리 중"):
    try:
        content = file_path.read_text(encoding="utf-8")
        source_url = _extract_source_url(content)
        tag = _get_api_tag_from_path(file_path)

        chunks = text_splitter.split_text(content)
        # 청크 단위 진행률
        for i, chunk in enumerate(
            tqdm(chunks, desc=f"{file_path.name} 청크 처리", leave=False)
        ):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": i,
                        "source": source_url,
                        "tags": tag,
                        "source_file": file_path.name,
                        "last_verified": today,
                    },
                )
            )
    except Exception as e:
        print(f"{file_path} 로드 중 오류 발생: {e}")

print(f"✅ 총 {len(documents)}개의 문서 청크 로드 완료")

BATCH_SIZE = 100  # 한 번에 삽입할 청크 개수

print("Document 삽입 중...")
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="DB 삽입 진행"):
    batch_docs = documents[i : i + BATCH_SIZE]
    collection.add(
        documents=[doc.page_content for doc in batch_docs],
        metadatas=[doc.metadata for doc in batch_docs],
        ids=[str(uuid.uuid4()) for _ in batch_docs],
    )

print(f"✅ 총 {len(documents)}개 Document 삽입 완료")
