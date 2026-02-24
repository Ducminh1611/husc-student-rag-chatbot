import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("⏳ Đang chuẩn bị dữ liệu...")

# Đảm bảo bạn đã để file chunking_file.json cùng thư mục với script này
with open("chunking_file.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

documents = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(raw_data)]

print("⏳ Đang load Model Embedding (E5)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"} # Đổi thành "cpu" nếu máy tính bạn không có card đồ họa rời
)

print("⏳ Đang tạo Vector Database (ChromaDB)...")
persist_dir = "./chroma_db"

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print(f"✅ Đã tạo xong Database tại thư mục: {persist_dir}")
print(f"✅ Tổng số chunk đã lưu: {len(documents)}")