import streamlit as st
import json
import torch
import gc
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================================
# 🎨 UI
# ==========================================================
st.set_page_config(page_title="RAG Hybrid Chatbot", page_icon="🎓", layout="wide")
st.title("Trợ lý Quy chế Sinh viên")

# ==========================================================
# 📂 LOAD RESOURCES
# ==========================================================
@st.cache_resource
def load_resources():
    status = st.sidebar.empty()
    status.info("⏳ Đang load tài nguyên...")

    with open("chunking_file.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    status.info("⏳ Kết nối ChromaDB...")
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda"} # Đổi thành "cpu" nếu không có card rời
    )

    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )

    status.info("⏳ Khởi tạo BM25...")
    tokenized = [doc.lower().split() for doc in raw_data]
    bm25 = BM25Okapi(tokenized)

    status.info("⏳ Load Reranker...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

    status.info("⏳ Load LLM Qwen 7B...")
    
    # LƯU Ý CHO BẠN: Sửa lại đường dẫn này. 
    # Nếu muốn máy tự tải model từ mạng, hãy đổi thành: MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True
    )

    status.success("✅ Sẵn sàng!")
    return raw_data, vector_db, bm25, reranker, tokenizer, model


raw_data, vector_db, bm25, reranker, tokenizer, model = load_resources()

# ==========================================================
# 🔍 HYBRID SEARCH
# ==========================================================
def hybrid_search(query, top_k=3):
    vec_docs = vector_db.similarity_search(query, k=10)
    vec_texts = [d.page_content for d in vec_docs]

    tokens = query.lower().split()
    bm25_texts = bm25.get_top_n(tokens, raw_data, n=10)

    candidates = list(set(vec_texts + bm25_texts))
    if not candidates:
        return []

    scores = reranker.predict([[query, c] for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [c[0] for c in ranked[:top_k]]

# ==========================================================
# 🤖 GENERATE RESPONSE
# ==========================================================
def generate_response(query):
    ctx = "\n\n".join(hybrid_search(query))

    messages = [
        {
            "role": "system",
            "content":
            "Bạn là **Trợ lý ảo dành cho sinh viên Trường Đại học Khoa học – Đại học Huế**.\n\n"
            "Nhiệm vụ của bạn:\n"
            "1. Luôn trả lời dựa trên **“Thông tin tham khảo”**.\n"
            "2. Chỉ sử dụng thông tin trong tài liệu.\n"
            "3. Nếu KHÔNG có thông tin, trả lời:\n"
            "“Xin lỗi, thông tin này không nằm trong tài liệu người dùng đã cung cấp. Tôi không thể trả lời.”\n"
            "4. Tuyệt đối không bịa.\n"
            "5. Tổng hợp chính xác, dễ hiểu.\n"
            "6. Không trả lời ngoài phạm vi tài liệu.\n"
        },
        {
            "role": "user",
            "content": f"Thông tin tham khảo:\n{ctx}\n\nCâu hỏi: {query}"
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.3)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.split("assistant\n")[-1]

# ==========================================================
# 💬 CHAT UI
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🧹 Xóa lịch sử"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            answer = generate_response(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    torch.cuda.empty_cache()
    gc.collect()