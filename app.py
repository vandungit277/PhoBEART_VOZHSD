import os
import time
import streamlit as st
import torch
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. CẤU HÌNH TRANG WEB & THEME
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Kiểm Duyệt Bình Luận - PhoBERT 1M WCE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) - Modern Tech Dashboard UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1040px;
    }

    /* Hero Banner Header */
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        color: #F8FAFC;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #93C5FD;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.25;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Action Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #CBD5E1;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        font-size: 1.05rem !important;
        height: 3.2em !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    }

    /* Result Panels */
    .result-panel-clean {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        color: #065F46;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.16);
    }
    .result-panel-hate {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        color: #991B1B;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.18);
    }

    /* Custom Progress Bars */
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .bar-container {
        width: 100%;
        background-color: #E2E8F0;
        border-radius: 9999px;
        height: 18px;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }
    .bar-fill-clean {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
    }
    .bar-fill-hate {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
    }

    /* Pipeline Monospace Box */
    .step-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        color: #0F172A;
        line-height: 1.5;
        word-break: break-word;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. KHỞI TẠO MÔ HÌNH & TOKENIZER (CACHED)
# ==========================================
def resolve_model_path():
    candidates = [
        "models/phobert_voz1m_weighted_ce",
        "Models/phobert_voz1m_weighted_ce",
        os.path.join(os.path.dirname(__file__), "models", "phobert_voz1m_weighted_ce"),
        os.path.join(os.path.dirname(__file__), "Models", "phobert_voz1m_weighted_ce"),
    ]
    for c in candidates:
        if os.path.isdir(c) and (os.path.exists(os.path.join(c, "model.safetensors")) or os.path.exists(os.path.join(c, "pytorch_model.bin"))):
            return c
    return candidates[0]

MODEL_PATH = resolve_model_path()

@st.cache_resource(show_spinner=False)
def load_phobert_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer load
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        
    # Model load
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

with st.spinner("⏳ Đang nạp mô hình PhoBERT (1M WCE) vào bộ nhớ..."):
    try:
        tokenizer, model, device = load_phobert_model(MODEL_PATH)
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception as e:
        st.error(f"❌ Không tìm thấy hoặc xảy ra lỗi khi nạp mô hình tại `{MODEL_PATH}`: {e}")
        st.stop()


# ==========================================
# 3. SIDEBAR: THÔNG TIN KỸ THUẬT & HỘI ĐỒNG
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ PhoBERT VOZ 1M")
    st.caption("Vietnamese Hate Speech Detection")
    st.divider()

    st.markdown("#### ⚙️ Thông số Môi trường")
    st.markdown(f"""
    - **Thiết bị**: `{device_name}`
    - **Chế độ**: `{'GPU (CUDA Acceleration)' if torch.cuda.is_available() else 'CPU (Inference Mode)'}`
    - **Phiên bản PyTorch**: `{torch.__version__}`
    - **Độ dài ngữ cảnh**: `max_length = 128`
    """)

    st.markdown("#### 🧠 Thông tin Mô hình")
    st.markdown("""
    - **Base Model**: `vinai/phobert-base`
    - **Tập dữ liệu**: **VOZ 1M Comments**
    - **Xử lý mất cân bằng**: *Class-Weighted Cross-Entropy (WCE)*
    - **Công cụ tách từ**: *PyVi (`ViTokenizer`)*
    """)

    st.divider()
    st.markdown("#### 🎯 Định nghĩa Nhãn")
    st.markdown("""
    - 🟢 **CLEAN (0)**: Bình luận chuẩn mực, thảo luận lịch sự, hợp lệ.
    - 🚫 **HATE (1)**: Bình luận công kích, thù hận, xúc phạm nhân phẩm.
    """)

    st.divider()
    if st.button("🗑️ Xoá lịch sử kiểm tra", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()


# ==========================================
# 4. GIAO DIỆN CHÍNH (HERO & INPUT)
# ==========================================
# Hero Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">🔬 Deep Learning NLP • Nghiên Cứu & Phát Triển</div>
    <h1 class="hero-title">HỆ THỐNG PHÁT HIỆN BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT</h1>
    <p class="hero-subtitle">Mô hình <strong>PhoBERT (1M Samples - Class-Weighted Cross-Entropy)</strong> kiểm duyệt và phân loại nội dung mạng xã hội theo thời gian thực.</p>
</div>
""", unsafe_allow_html=True)

# Khởi tạo Session State
if "comment_input" not in st.session_state:
    st.session_state["comment_input"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []
if "auto_run" not in st.session_state:
    st.session_state["auto_run"] = False

# Quick Presets (One-Click Test)
st.markdown("##### 💡 Thử nhanh với các câu mẫu (One-Click Test):")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

preset_1 = "Sản phẩm xài rất thích, giao hàng cực nhanh!"
preset_2 = "đi ăn cháo óc heo không"
preset_3 = "Mấy thằng ngu này làm ăn như lừa đảo."
preset_4 = "ĐM!"

if col_p1.button("🟢 Mẫu 1 (Sạch)", help=preset_1, use_container_width=True):
    st.session_state["comment_input"] = preset_1
    st.session_state["auto_run"] = True
    st.rerun()

if col_p2.button("🟢 Mẫu 2 (Sạch)", help=preset_2, use_container_width=True):
    st.session_state["comment_input"] = preset_2
    st.session_state["auto_run"] = True
    st.rerun()

if col_p3.button("🚫 Mẫu 3 (Độc hại)", help=preset_3, use_container_width=True):
    st.session_state["comment_input"] = preset_3
    st.session_state["auto_run"] = True
    st.rerun()

if col_p4.button("🚫 Mẫu 4 (Độc hại)", help=preset_4, use_container_width=True):
    st.session_state["comment_input"] = preset_4
    st.session_state["auto_run"] = True
    st.rerun()

# Text Area
user_comment = st.text_area(
    "Nhập bình luận tiếng Việt cần kiểm duyệt tại đây:",
    key="comment_input",
    height=110,
    placeholder="Nhập chuỗi bình luận tiếng Việt cần kiểm duyệt tại đây..."
)

# Nút hành động
c_btn_run, c_btn_reset = st.columns([4, 1])
with c_btn_run:
    manual_run = st.button("🔍 BẮT ĐẦU KIỂM TRA", type="primary", use_container_width=True)
with c_btn_reset:
    if st.button("🔄 Làm mới", use_container_width=True):
        st.session_state["comment_input"] = ""
        st.session_state["auto_run"] = False
        st.rerun()

should_infer = manual_run or st.session_state["auto_run"]
if st.session_state["auto_run"]:
    st.session_state["auto_run"] = False


# ==========================================
# 5. PIPELINE XỬ LÝ & SUY LUẬN
# ==========================================
if should_infer:
    active_text = user_comment.strip()
    
    if not active_text:
        st.warning("⚠️ Vui lòng nhập nội dung câu bình luận trước khi kiểm tra!")
    else:
        t_start = time.perf_counter()
        
        # Bước 1: Lowercase (Chuẩn hóa chữ thường)
        comment_clean = active_text.lower()
        
        # Bước 2: Tách từ bằng PyVi
        segmented_text = ViTokenizer.tokenize(comment_clean)
        
        # Bước 3: Mã hóa với Tokenizer
        inputs = tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        inputs_device = {k: v.to(device) for k, v in inputs.items()}
        
        # Bước 4: Inference & Softmax
        with torch.no_grad():
            outputs = model(**inputs_device)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            prob_clean = probs[0].item() * 100.0
            prob_hate = probs[1].item() * 100.0
            pred_label = torch.argmax(probs).item()
            
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        
        # Lưu vào lịch sử phiên làm việc
        st.session_state["history"].insert(0, {
            "time": time.strftime("%H:%M:%S"),
            "text": active_text,
            "label": "HATE" if pred_label == 1 else "CLEAN",
            "prob_clean": prob_clean,
            "prob_hate": prob_hate,
            "ms": t_elapsed_ms
        })
        
        st.divider()

        # ==========================================
        # 6. HIỂN THỊ KẾT QUẢ (STATUS & GAUGES)
        # ==========================================
        st.markdown("### 📊 Kết Quả Phân Tích")

        if pred_label == 1:
            st.markdown(f"""
            <div class="result-panel-hate">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <span style="font-size: 1.85rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                            🚫 PHÁT HIỆN: HATE (Bình luận Độc hại / Thù hận)
                        </span>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.95;">
                            Nội dung chứa yếu tố xúc phạm danh dự, kỳ thị, thù hận hoặc tiêu cực nghiêm trọng.
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 2.3rem; font-weight: 800;">{prob_hate:.2f}%</span>
                        <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">Độ tin cậy HATE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-panel-clean">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <span style="font-size: 1.85rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                            🟢 PHÁT HIỆN: CLEAN (Bình thường / An toàn)
                        </span>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.95;">
                            Nội dung bình luận lành mạnh, chuẩn mực và an toàn với cộng đồng mạng xã hội.
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 2.3rem; font-weight: 800;">{prob_clean:.2f}%</span>
                        <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">Độ tin cậy CLEAN</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Thanh đo độ tin cậy trực quan
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="bar-label">
                <span>🟢 Lớp CLEAN (An toàn / Lành mạnh)</span>
                <span style="color: #059669;">{prob_clean:.2f}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill-clean" style="width: {prob_clean:.2f}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="bar-label">
                <span>🚫 Lớp HATE (Độc hại / Thù hận)</span>
                <span style="color: #DC2626;">{prob_hate:.2f}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill-hate" style="width: {prob_hate:.2f}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        # Huy hiệu thông số suy luận
        real_tokens = int((inputs['attention_mask'][0] == 1).sum().item())
        st.markdown(f"""
        <div style="margin-bottom: 1.2rem;">
            <span class="metric-badge">⚡ Thời gian suy luận: <strong>{t_elapsed_ms:.2f} ms</strong></span>
            <span class="metric-badge">💻 Thiết bị: <strong>{device_name}</strong></span>
            <span class="metric-badge">📏 Ngữ cảnh: <strong>128 tokens</strong></span>
            <span class="metric-badge">🔤 Số tokens thực tế: <strong>{real_tokens}</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # ==========================================
        # 7. CHI TIẾT PIPELINE TIỀN XỬ LÝ (ACCORDION)
        # ==========================================
        with st.expander("🔍 Xem chi tiết Pipeline tiền xử lý NLP (Dành cho Báo cáo / Hội đồng)", expanded=False):
            st.markdown("Quy trình 4 bước xử lý văn bản theo đúng chuẩn huấn luyện mô hình PhoBERT:")
            
            st.markdown("**1. Văn bản gốc (Raw Input):**")
            st.markdown(f'<div class="step-box">{active_text}</div>', unsafe_allow_html=True)
            
            st.markdown("**2. Chuẩn hóa chữ thường (Lowercase):**")
            st.markdown(f'<div class="step-box">{comment_clean}</div>', unsafe_allow_html=True)
            
            st.markdown("**3. Tách từ tiếng Việt bằng PyVi (`ViTokenizer.tokenize`):**")
            st.markdown(f'<div class="step-box">{segmented_text}</div>', unsafe_allow_html=True)
            st.caption("ℹ️ *PyVi tự động ghép các từ phức/thành ngữ bằng dấu gạch dưới `_` giúp PhoBERT ánh xạ chính xác token.*")
            
            st.markdown("**4. Mã hóa đầu vào (Tokenization & Tensor Representation):**")
            token_ids_preview = inputs['input_ids'][0][:min(real_tokens, 24)].tolist()
            st.markdown(f"""
            - **Độ dài chuỗi token (không tính padding)**: `{real_tokens}`
            - **Kích thước Tensor `input_ids`**: `{list(inputs['input_ids'].shape)}`
            - **Kích thước Tensor `attention_mask`**: `{list(inputs['attention_mask'].shape)}`
            - **Mẫu 24 Token IDs đầu tiên**: `{token_ids_preview}`
            """)


# ==========================================
# 8. LỊCH SỬ THỬ NGHIỆM TRONG PHIÊN
# ==========================================
if st.session_state["history"]:
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🕒 Lịch Sử Thử Nghiệm Gần Đây")
    
    for item in st.session_state["history"][:6]:
        is_clean = item["label"] == "CLEAN"
        border_col = "#10B981" if is_clean else "#EF4444"
        badge_txt = "🟢 CLEAN" if is_clean else "🚫 HATE"
        score = item["prob_clean"] if is_clean else item["prob_hate"]
        
        st.markdown(f"""
        <div style="background: rgba(248, 250, 252, 0.9); border: 1px solid #E2E8F0; border-left: 4px solid {border_col}; border-radius: 8px; padding: 10px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div style="max-width: 72%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                <span style="color: #64748B; font-size: 0.8rem; margin-right: 8px;">[{item['time']}]</span>
                <span style="font-weight: 500; color: #1E293B;">"{item['text']}"</span>
            </div>
            <div>
                <span style="background: {border_col}18; color: {border_col}; border: 1px solid {border_col}44; padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 700;">
                    {badge_txt} ({score:.2f}%)
                </span>
                <span style="color: #94A3B8; font-size: 0.75rem; margin-left: 8px;">{item['ms']:.1f}ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
