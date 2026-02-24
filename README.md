# husc-student-rag-chatbot
# 📖 RAG Chatbot – Trợ lý Ảo Tư Vấn Quy Chế Sinh Viên HUSC
**(Retrieval-Augmented Generation for Academic Regulations QA)**

Dự án này xây dựng một hệ thống Retrieval-Augmented Generation (RAG) nhằm hỗ trợ sinh viên tra cứu Sổ tay Sinh viên và Quy chế Đào tạo của Trường Đại học Khoa học – Đại học Huế bằng ngôn ngữ tự nhiên.

---

## 🌟 Giới thiệu
Hệ thống được thiết kế để giải quyết bài toán tra cứu văn bản quy phạm pháp lý phức tạp, đảm bảo:
* **Trả lời đúng ngữ cảnh:** Trích dẫn đúng điều khoản, chương mục.
* **Minh bạch:** Có trích dẫn nguồn rõ ràng từ tài liệu gốc.
* **Zero Hallucination:** Không bịa đặt thông tin; chủ động từ chối trả lời khi tài liệu không chứa thông tin phù hợp.


---

## ✨ Điểm nổi bật (Key Features)

### 🧩 Semantic Chunking & Header Injection
* Văn bản không bị cắt theo độ dài cố định mà được chia theo cấu trúc pháp lý: **Chương → Mục → Điều → Khoản**.
* Tự động gắn tiêu đề ngữ cảnh (Header Injection) vào từng chunk để giữ trọn ý nghĩa và tránh nhầm lẫn điều khoản giữa các chương.

### 🔍 Hybrid Retrieval (Truy hồi lai)
Kết hợp hai phương pháp để tối ưu hóa kết quả:
* **Sparse Retrieval (BM25):** Mạnh trong việc tìm kiếm từ khóa chính xác và số hiệu điều khoản.
* **Dense Retrieval (Vector Search):** Hiểu được ý nghĩa ngữ nghĩa của các câu hỏi tự nhiên từ sinh viên.

### ⚖️ Cross-Encoder Re-ranking
* Các đoạn văn bản sau khi truy hồi được chấm điểm lại bằng Re-ranker (`bge-reranker-v2-m3`).
* Chỉ những nội dung liên quan nhất mới được đưa vào mô hình sinh (LLM), giúp tăng độ chính xác và giảm nhiễu.

### 🛡️ Refusal Mechanism
* Hệ thống chỉ trả lời dựa trên ngữ cảnh đã được truy hồi.
* Nếu điểm truy hồi dưới ngưỡng an toàn, hệ thống sẽ từ chối trả lời để tránh hiện tượng ảo giác (hallucination).

---

## 🛠 Kiến trúc Công nghệ (Tech Stack)

Hệ thống được tối ưu để chạy trên môi trường tài nguyên giới hạn như **Google Colab** hoặc **GPU T4**.

| Thành phần | Công nghệ |
| :--- | :--- |
| **Embedding Model** | `intfloat/multilingual-e5-large` |
| **Vector Database** | `ChromaDB` |
| **Sparse Retrieval** | `BM25 (rank_bm25)` |
| **Re-ranking Model** | `BAAI/bge-reranker-v2-m3` |
| **Large Language Model** | `Qwen2.5-Instruct-7B` (Quantization 4-bit) |
| **UI Framework** | `Streamlit` |

---

## 📊 Đánh giá Hệ thống (RAGAS Evaluation)

Kết quả đánh giá trên 50 câu hỏi kiểm thử thực tế:

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| **Faithfulness** | **0.7744** | Mức độ trung thực, hạn chế bịa đặt thông tin. |
| **Answer Relevancy** | **0.7989** | Mức độ phù hợp của câu trả lời với câu hỏi. |
| **Answer Correctness** | **0.7457** | Độ chính xác về nội dung kiến thức. |
| **Context Precision** | **0.6966** | Độ chính xác của ngữ cảnh được trích xuất. |
| **Context Recall** | **0.7497** | Mức độ bao phủ đầy đủ thông tin cần thiết. |

---

## 🚀 Hướng dẫn Cài đặt và Chạy

### 1. Clone repository
```bash
git clone [https://github.com/Ducminh1611/husc-student-rag-chatbot.git](https://github.com/Ducminh1611/husc-student-rag-chatbot.git)
cd husc-student-rag-chatbot


### 2. Cài đặt thư viện
Cài đặt các gói phụ thuộc cần thiết (PyTorch, Transformers, LangChain, v.v.):
```bash
pip install -r requirements.txt
Bash
python build_db.py


###3. Chạy ứng dụng Streamlit
```Bash
streamlit run app.py
Yêu cầu Phần cứng
GPU khuyến nghị: Tối thiểu 8GB VRAM (để chạy Qwen2.5-7B 4-bit).
RAM: Tối thiểu 12GB.
Tác giả:

Nguyễn Hồng Sơn
Phạm Văn Quân
Đỗ Văn Sinh
Nguyễn Đức Minh

## 📚 Tài liệu tham khảo (References)

Hệ thống được xây dựng dựa trên các nghiên cứu và công nghệ tiên tiến trong lĩnh vực NLP và Hệ thống truy hồi:

1.  **Lewis, P., et al.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2.  **Izacard, G., & Grave, E.** (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*. NeurIPS.
3.  **Robertson, S., & Zaragoza, H.** (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
4.  **Karpukhin, V., et al.** (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
5.  **BAAI.** (2024). *BGE-M3: Multi-lingual, Multi-granularity Text Embeddings*. arXiv.
6.  **Reimers, N., & Gurevych, I.** (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
7.  **Cormack, G. V., Clarke, C. L. A.** (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR.
8.  **Nogueira, R., & Cho, K.** (2019). *Passage Re-ranking with BERT*. arXiv.