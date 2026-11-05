# YÊU CẦU PHÁT TRIỂN CÔNG CỤ WEB DEMO PHÂN LOẠI BÌNH LUẬN ĐỘC HẠI (PHOBERT 1M WCE)
> **Dành cho**: Agent Antigravity / Coding Assistant  
> **Mục tiêu**: Xây dựng giao diện Web UI hiện đại, đẹp mắt và trực quan để thuyết trình Demo bài tập lớn/đề tài nghiên cứu khoa học.

---

## 📌 1. BỐI CẢNH & MỤC TIÊU PROJECT

Bạn là một Chuyên gia Lập trình Web và UI/UX Designer. Hãy giúp tôi tạo một ứng dụng Web Demo có giao diện hiện đại, chuyên nghiệp, mượt mà và thân thiện với người dùng để thay thế giao diện dạng Console Terminal (Colab).

Ứng dụng Web này sẽ nạp mô hình Học sâu **PhoBERT** đã được fine-tune để phát hiện bình luận độc hại/thù hận tiếng Việt và đưa ra phân tích thời gian thực.

---

## 🛠️ 2. ĐƯỜNG DẪN MÔ HÌNH & THÔNG TIN KỸ THUẬT

* **Đường dẫn thư mục mô hình local**: `models/phobert_voz1m_weighted_ce`
* **Kiến trúc mô hình**: `AutoModelForSequenceClassification` (PhoBERT 1M Class-Weighted CE).
* **Các thư viện Python cần dùng**:
  * `torch`
  * `transformers`
  * `pyvi` (`ViTokenizer`)
  * `sentencepiece`
  * Framework Web UI: **Streamlit** (khuyên dùng để dựng UI siêu đẹp, nhanh) hoặc **Gradio** / **FastAPI + HTML/Tailwind CSS**.

---

## ⚙️ 3. PIPELINE TIỀN XỬ LÝ DỮ LIỆU (BẮT BUỘC ĐÚNG THỨ TỰ)

Khi người dùng nhập chuỗi bình luận vào Web UI, ứng dụng phải xử lý theo đúng 4 bước sau:

1. **Lowercase (Chuẩn hóa chữ thường)**: `comment_clean = comment.lower()`
2. **Word Segmentation (Tách từ Tiếng Việt bằng PyVi)**: `segmented_text = ViTokenizer.tokenize(comment_clean)`
3. **Tokenization (Mã hóa văn bản)**:
   ```python
   inputs = tokenizer(
       segmented_text,
       return_tensors="pt",
       truncation=True,
       max_length=128,
       padding="max_length"
   )
   ```
4. **Inference & Softmax (Suy luận)**:
   * Chuyển Tensor sang GPU/CPU thiết bị khả thi (`cuda` nếu có, ngược lại `cpu`).
   * Tính xác suất qua hàm `softmax(outputs.logits, dim=-1)`.
   * Lấy `% CLEAN` (lớp 0) và `% HATE` (lớp 1).
   * Dự đoán nhãn:
     * **Nhãn 0**: `🟢 CLEAN (Bình thường / An toàn)`
     * **Nhãn 1**: `🚫 HATE (Thù hận / Độc hại / Xúc phạm)`

---

## 🎨 4. YÊU CẦU VỀ GIAO DIỆN VÀ TRẢI NGHIỆM NGƯỜI DÙNG (UI/UX)

1. **Phong cách Thiết kế (Theme)**:
   * Giao diện hiện đại (Modern Tech Dashboard), hỗ trợ Dark Mode hoặc Tone màu trung tính sang trọng (Deep Navy, Dark Slate, hoặc Light Clean).
   * Tiêu đề chính nổi bật với Icon và Subtitle chuyên nghiệp:  
     **"HỆ THỐNG PHÁT HIỆN BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT"**  
     *Mô hình PhoBERT (1M Samples - Class-Weighted Cross-Entropy)*.

2. **Khu vực Nhập dữ liệu (Input Section)**:
   * Khung nhập văn bản lớn (`TextArea`) cho phép gán placeholder mượt mà: *"Nhập bình luận tiếng Việt cần kiểm duyệt tại đây..."*.
   * Nút bấm **"🔍 BẮT ĐẦU KIỂM TRA"** nổi bật với hiệu ứng nhấn.
   * **Bộ nút mẫu thử nhanh (Quick Presets / One-Click Test)**: Tạo 4 nút bấm mẫu để giảng viên/hội đồng có thể click chạy ngay mà không cần tự gõ:
     * 🟢 *Mẫu 1 (Sạch)*: `"Sản phẩm xài rất thích, giao hàng cực nhanh!"`
     * 🟢 *Mẫu 2 (Sạch)*: `"đi ăn cháo óc heo không"`
     * 🚫 *Mẫu 3 (Độc hại)*: `"Mấy thằng ngu này làm ăn như lừa đảo."`
     * 🚫 *Mẫu 4 (Độc hại)*: `"ĐM!"`

3. **Khu vực Hiển thị Kết quả (Output Section)**:
   * **Thẻ trạng thái (Status Card)**: Tự động đổi màu nền nổi bật theo kết quả:
     * Nền Xanh Lá (`Green Alert Panel`) nếu là **CLEAN**.
     * Nền Đỏ/Cam (`Red Alert Panel`) nếu là **HATE**.
   * **Thanh đo độ tin cậy (Confidence Gauges / Progress Bars)**:
     * Biểu diễn trực quan bằng 2 thanh phần trăm (%) sinh động cho `% CLEAN` và `% HATE`.
   * **Khung xem chi tiết Pipeline tiền xử lý (Pipeline Accordion)**:
     * Cho phép mở rộng (Expander) để soi bước xử lý trung gian:
       * *Text gốc*: `"Mấy thằng ngu này"`
       * *Text Lowercase*: `"mấy thằng ngu này"`
       * *Text qua PyVi*: `"mấy thằng ngu này"` (nối từ `ngu_này` nếu có)

---

## 💻 5. MÃ NGUỒN MẪU GỢI Ý (STREAMLIT FRAMEWORK)

Hãy tạo cấu trúc thư mục project như sau:

```text
my_project/
├── models/
│   └── phobert_voz1m_weighted_ce/   <-- chứa model & tokenizer đã train
├── app.py                            <-- File chạy giao diện chính Streamlit
└── requirements.txt                  <-- Thư viện phụ thuộc
```

### Nội dung file `app.py` mẫu:

```python
import streamlit as st
import torch
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(
    page_title="Phát Hiện Bình Luận Độc Hại - PhoBERT",
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cache & Load Model
MODEL_PATH = "models/phobert_voz1m_weighted_ce"

@st.cache_resource
def load_phobert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_phobert_model()

# Header
st.markdown("<div class='main-title'>🛡️ HỆ THỐNG KIỂM DUYỆT BÌNH LUẬN</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Mô hình PhoBERT (1M Samples - Class-Weighted Cross-Entropy)</div>", unsafe_allow_html=True)

# Session state for quick input
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Quick Presets
st.write("💡 **Thử nhanh với các câu mẫu:**")
col1, col2, col3, col4 = st.columns(4)

if col1.button("🟢 Mẫu sạch 1"):
    st.session_state["input_text"] = "Sản phẩm xài rất thích, giao hàng cực nhanh!"
if col2.button("🟢 Mẫu sạch 2"):
    st.session_state["input_text"] = "đi ăn cháo óc heo không"
if col3.button("🚫 Mẫu độc hại 1"):
    st.session_state["input_text"] = "Mấy thằng ngu này làm ăn như lừa đảo."
if col4.button("🚫 Mẫu độc hại 2"):
    st.session_state["input_text"] = "ĐM!"

# Input Area
user_comment = st.text_area(
    "Nhập nội dung bình luận cần kiểm tra:",
    value=st.session_state["input_text"],
    height=100,
    placeholder="Nhập chuỗi văn bản tiếng Việt..."
)

# Inference Function
if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN", type="primary"):
    if not user_comment.strip():
        st.warning("⚠️ Vui lòng nhập nội dung câu bình luận!")
    else:
        # Pipeline
        comment_clean = user_comment.lower()
        segmented_text = ViTokenizer.tokenize(comment_clean)
        
        inputs = tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            prob_clean = probs[0].item() * 100
            prob_hate = probs[1].item() * 100
            pred_label = torch.argmax(probs).item()
        
        st.divider()
        
        # Output Visuals
        if pred_label == 1:
            st.error(f"### 🚫 KẾT QUẢ: HATE (Bình luận Độc hại / Thù hận)\n**Độ tin cậy lớp HATE**: {prob_hate:.2f}%")
        else:
            st.success(f"### 🟢 KẾT QUẢ: CLEAN (Bình thường / An toàn)\n**Độ tin cậy lớp CLEAN**: {prob_clean:.2f}%")
        
        # Progress Bars
        st.write("📊 **Xác suất phân rã từng lớp:**")
        st.write(f"**CLEAN (An toàn):** {prob_clean:.2f}%")
        st.progress(int(prob_clean))
        
        st.write(f"**HATE (Độc hại):** {prob_hate:.2f}%")
        st.progress(int(prob_hate))
        
        # Expander for Pipeline Details
        with st.expander("🔍 Xem chi tiết các bước tiền xử lý NLP"):
            st.write(f"- **Văn bản gốc**: `{user_comment}`")
            st.write(f"- **Sau Lowercase**: `{comment_clean}`")
            st.write(f"- **Sau PyVi Tokenize**: `{segmented_text}`")
```

---

## 🚀 6. HƯỚNG DẪN THỰC THI

1. Tạo file `requirements.txt`:
   ```text
   streamlit
   torch
   transformers
   pyvi
   sentencepiece
   ```
2. Chạy ứng dụng web:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

---
*Hãy hoàn thiện ứng dụng Web này theo phong cách thiết kế hiện đại nhất có thể để phục vụ buổi thuyết trình Demo ấn tượng!*
