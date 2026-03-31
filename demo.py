import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ================== CẤU HÌNH TRANG ==================
st.set_page_config(
    page_title="Nội Suy Đa Thức - Đề Tài 4",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS ==================
st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; text-align: center; padding: 10px 0; }
    .section-header { background: linear-gradient(90deg, #16213e, #0f3460); color: white;
        padding: 8px 16px; border-radius: 6px; margin: 16px 0 8px 0; font-size: 1.05rem; font-weight: 600; }
    .formula-box { background: #f0f4ff; border-left: 4px solid #4c7ef3;
        padding: 12px 16px; border-radius: 4px; font-family: monospace; font-size: 0.92rem; }
    .metric-good { color: #2ecc71; font-weight: bold; }
    .metric-warn { color: #e67e22; font-weight: bold; }
    .metric-bad  { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================== THUẬT TOÁN ==================

def lagrange(x_data, y_data, x):
    """Nội suy Lagrange - đa thức dạng tổng tích Lagrange basis."""
    n = len(x_data)
    if len(set(x_data)) < n:
        return None
    result = 0.0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                denom = x_data[i] - x_data[j]
                if abs(denom) < 1e-14:
                    return None
                term *= (x - x_data[j]) / denom
        result += term
    return result


def divided_difference(x, y):
    """Xây dựng bảng tỷ sai phân (cho mốc không cách đều)."""
    n = len(x)
    if len(set(x)) < n:
        return None
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = float(y[i])
    for j in range(1, n):
        for i in range(n - j):
            denom = x[i + j] - x[i]
            if abs(denom) < 1e-14:
                return None
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / denom
    return table


def newton_unequal(x_data, table, x):
    """Newton nội suy (mốc KHÔNG cách đều) - dùng bảng tỷ sai phân."""
    n = len(x_data)
    result = table[0][n - 1]
    for i in range(n - 2, -1, -1):
        result = table[0][i] + (x - x_data[i]) * result
    return result


def forward_diff_table(y):
    """Bảng sai phân tiến (cho mốc cách đều)."""
    n = len(y)
    table = [list(map(float, y))]
    for _ in range(1, n):
        prev = table[-1]
        row = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
        if not row:
            break
        table.append(row)
    return table


def newton_forward(x_data, y_data, x):
    """Newton nội suy tiến (mốc CÁCH ĐỀU)."""
    n = len(x_data)
    h = x_data[1] - x_data[0]
    if abs(h) < 1e-14:
        return None
    s = (x - x_data[0]) / h
    table = forward_diff_table(y_data)
    result = table[0][0]
    s_term = 1.0
    factorial = 1
    for k in range(1, n):
        if k >= len(table) or not table[k]:
            break
        s_term *= (s - (k - 1))
        factorial *= k
        result += s_term / factorial * table[k][0]
    return result


def is_equally_spaced(x, tol=1e-9):
    """Kiểm tra mốc có cách đều không."""
    if len(x) < 2:
        return True
    h = x[1] - x[0]
    return all(abs((x[i + 1] - x[i]) - h) < tol for i in range(len(x) - 1))


def get_k_points(target_x, valid_x, valid_y, k):
    """Lấy k điểm gần nhất với target_x."""
    data = sorted(zip(valid_x, valid_y), key=lambda item: item[0])
    idx = min(range(len(data)), key=lambda i: abs(data[i][0] - target_x))
    left = max(0, idx - k // 2)
    right = min(len(data), left + k)
    if right - left < k:
        left = max(0, right - k)
    selected = data[left:right]
    return [p[0] for p in selected], [p[1] for p in selected]


# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("## ⚙️ Thiết lập")

    st.markdown("### 📂 Dữ liệu đầu vào")
    input_mode = st.radio("Nguồn dữ liệu", ["Upload CSV (có NaN)", "Tạo dữ liệu mẫu"])

    uploaded_file = None
    if input_mode == "Upload CSV (có NaN)":
        uploaded_file = st.file_uploader("Tải lên file CSV (có thể chứa NaN)", type=["csv"])
        st.caption("💡 File cần có ít nhất 1 cột số với một số ô trống (NaN).")
    else:
        demo_type = st.selectbox("Chọn loại dữ liệu mẫu", [
            "Nhiệt độ theo ngày (có NaN)",
            "Giá cổ phiếu (có NaN)",
            "Hàm sin + noise (có NaN)"
        ])
        n_points = st.slider("Số điểm dữ liệu", 15, 60, 30)
        nan_ratio = st.slider("Tỷ lệ NaN có sẵn trong dữ liệu (%)", 5, 40, 20)
        if st.button("🎲 Tạo dữ liệu mẫu"):
            np.random.seed(42)
            x = np.arange(n_points, dtype=float)
            if "Nhiệt độ" in demo_type:
                y = 20 + 5 * np.sin(x * 0.3) + np.random.normal(0, 0.8, n_points)
            elif "cổ phiếu" in demo_type:
                y = np.cumsum(np.random.normal(0.2, 1.5, n_points)) + 100
            else:
                y = np.sin(x * 0.5) + 0.1 * np.random.randn(n_points)
            y = y.astype(float)
            nan_idx = random.sample(range(2, n_points - 2), int(n_points * nan_ratio / 100))
            for i in nan_idx:
                y[i] = np.nan
            st.session_state["data"] = pd.DataFrame({"x": x, "y": y})
            st.success(f"✅ Đã tạo {n_points} điểm với {len(nan_idx)} NaN")

    st.markdown("---")
    st.markdown("### 🔬 Phương pháp nội suy")

    method_raw = st.radio("Phương pháp", [
        "Lagrange",
        "Newton (không cách đều)",
        "Newton (cách đều - sai phân tiến)",
        "So sánh Lagrange & Newton"
    ])

    mode_display = st.radio("Chế độ lựa chọn mốc", [
        "Lân cận (k điểm gần nhất)",
        "Toàn cục (dùng toàn bộ dữ liệu)"
    ])
    mode = "Local" if "Lân cận" in mode_display else "Global"
    k = st.slider("Số điểm k dùng để nội suy", 3, 15, 5)

    st.markdown("---")
    st.markdown("### 🎯 Phạm vi nội suy")
    apply_range = st.checkbox("Giới hạn khoảng X nội suy", value=False)
    run_btn = st.button("🚀 Thực hiện nội suy", type="primary", use_container_width=True)

# ================== LOAD DỮ LIỆU ==================
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # Chuẩn hóa NaN
        df = df.replace(["", " ", "NA", "N/A", "nan", "NaN", "null", "NULL"], np.nan)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file CSV: {e}")
        st.stop()
elif "data" in st.session_state:
    df = st.session_state["data"]

if df is None:
    st.markdown('<div class="main-title">📈 Demo Nội Suy Đa Thức - Xử Lý Dữ Liệu Khuyết</div>', unsafe_allow_html=True)
    st.info("👈 Hãy **upload file CSV** hoặc **tạo dữ liệu mẫu** từ thanh bên để bắt đầu.")
    
    st.markdown('<div class="section-header">📘 Hướng dẫn nhanh</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Dùng file CSV của bạn:**
- File có cột X (số thực) và cột Y
- Các ô trống / NaN trong cột Y sẽ được nội suy
- Ví dụ: dữ liệu nhiệt độ, cảm biến, tài chính...
        """)
    with col2:
        st.markdown("""
**Tính năng chính:**
- ✅ Nội suy Lagrange tự viết
- ✅ Newton không cách đều (tỷ sai phân)
- ✅ Newton cách đều (sai phân tiến)
- ✅ So sánh hiệu năng & sai số MAE
- ✅ Phân tích hiện tượng Runge
        """)
    st.stop()

# ================== HIỂN THỊ DỮ LIỆU ==================
st.markdown('<div class="main-title">📈 Nội Suy Đa Thức - Xử Lý Dữ Liệu Khuyết</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">📊 Dữ liệu đầu vào</div>', unsafe_allow_html=True)

col_data1, col_data2 = st.columns([3, 1])
with col_data1:
    st.dataframe(df, use_container_width=True, height=220)
with col_data2:
    st.metric("Tổng hàng", len(df))
    total_nan = df.isnull().sum().sum()
    st.metric("Ô NaN", int(total_nan))
    st.metric("Số cột", len(df.columns))

with st.expander("⚙️ Chọn cột X / Y"):
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Cột X (trục ngang)", df.columns)
    with c2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y_col = st.selectbox("Cột Y (cần nội suy)", numeric_cols if numeric_cols else df.columns)

# Tự động xử lý cột X: nếu là ngày/text thì chuyển sang số thứ tự
try:
    x_raw = pd.to_numeric(df[x_col], errors="coerce")
    if x_raw.isna().sum() > len(df) * 0.3:
        # Cột X có quá nhiều NaN sau to_numeric → đây là cột ngày hoặc text
        # Chuyển sang số thứ tự 1, 2, 3, ...
        x_all = list(range(1, len(df) + 1))
        x_labels = df[x_col].tolist()  # Giữ nhãn gốc để hiển thị
        st.info(f"💡 Cột **{x_col}** chứa ngày/text → tự động chuyển sang số thứ tự (1, 2, 3, ...)")
    else:
        x_all = x_raw.tolist()
        x_labels = x_all
    y_all_raw = pd.to_numeric(df[y_col], errors="coerce").tolist()
except Exception as e:
    st.error(f"Lỗi đọc cột dữ liệu: {e}")
    st.stop()

# Phạm vi X
min_x_val = float(np.nanmin(x_all))
max_x_val = float(np.nanmax(x_all))

if apply_range:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        target_min = st.number_input("X bắt đầu", value=min_x_val, min_value=min_x_val, max_value=max_x_val)
    with col_r2:
        target_max = st.number_input("X kết thúc", value=max_x_val, min_value=min_x_val, max_value=max_x_val)
else:
    target_min, target_max = min_x_val, max_x_val

# ================== THỰC HIỆN NỘI SUY ==================
if run_btn:
    y_true = y_all_raw.copy()
    y_work = y_all_raw.copy()

    # Các điểm hợp lệ (không NaN)
    valid_pairs = [(x_all[i], y_work[i]) for i in range(len(x_all))
                   if x_all[i] is not None and not math.isnan(x_all[i])
                   and y_work[i] is not None and not math.isnan(y_work[i])]

    if len(valid_pairs) < 3:
        st.error("❌ Cần ít nhất 3 điểm không khuyết để nội suy. Hãy kiểm tra lại dữ liệu.")
        st.stop()

    valid_x = [p[0] for p in valid_pairs]
    valid_y = [p[1] for p in valid_pairs]

    # Các vị trí cần nội suy (NaN trong phạm vi)
    missing_idx = [
        i for i in range(len(x_all))
        if (y_work[i] is None or math.isnan(float(y_work[i])))
        and x_all[i] is not None and not math.isnan(x_all[i])
        and target_min <= x_all[i] <= target_max
    ]

    if not missing_idx:
        st.warning(f"⚠️ Không có điểm NaN nào trong khoảng X = [{target_min:.2f}, {target_max:.2f}].")
        st.stop()

    if any(i < k or i > len(x_all) - k for i in missing_idx):
        st.warning("⚠️ Một số điểm khuyết gần biên — nội suy có thể kém chính xác.")

    # Xác định loại phương pháp
    use_lagrange = "Lagrange" in method_raw or "So sánh" in method_raw
    use_newton_unequal = ("không cách đều" in method_raw) or "So sánh" in method_raw
    use_newton_equal = "cách đều" in method_raw and "không cách đều" not in method_raw

    pred_lag = y_work.copy()
    pred_new = y_work.copy()
    time_lag = time_new = 0.0
    mae_lag = mae_new = None

    # ===== LAGRANGE =====
    if use_lagrange:
        t0 = time.perf_counter()
        for i in missing_idx:
            nx, ny = (get_k_points(x_all[i], valid_x, valid_y, k)
                      if mode == "Local" else (valid_x, valid_y))
            val = lagrange(nx, ny, x_all[i])
            pred_lag[i] = val if val is not None else float("nan")
        time_lag = time.perf_counter() - t0

    # ===== NEWTON KHÔNG CÁCH ĐỀU =====
    if use_newton_unequal:
        t0 = time.perf_counter()
        if mode == "Global":
            global_table = divided_difference(valid_x, valid_y)
        for i in missing_idx:
            if mode == "Local":
                nx, ny = get_k_points(x_all[i], valid_x, valid_y, k)
                table = divided_difference(nx, ny)
            else:
                nx, ny = valid_x, valid_y
                table = global_table
            if table is not None:
                pred_new[i] = newton_unequal(nx, table, x_all[i])
            else:
                pred_new[i] = float("nan")
        time_new = time.perf_counter() - t0

    # ===== NEWTON CÁCH ĐỀU =====
    if use_newton_equal:
        t0 = time.perf_counter()
        for i in missing_idx:
            if mode == "Local":
                nx, ny = get_k_points(x_all[i], valid_x, valid_y, k)
            else:
                nx, ny = valid_x, valid_y
            if is_equally_spaced(nx):
                val = newton_forward(nx, ny, x_all[i])
            else:
                # fallback về tỷ sai phân nếu không cách đều
                table = divided_difference(nx, ny)
                val = newton_unequal(nx, table, x_all[i]) if table else float("nan")
            pred_new[i] = val if val is not None else float("nan")
        time_new = time.perf_counter() - t0

    # ===== TÍNH SAI SỐ =====
    def calc_mae(pred):
        errors = [abs(pred[i] - y_true[i])
                  for i in missing_idx
                  if y_true[i] is not None and not math.isnan(float(y_true[i]))
                  and pred[i] is not None and not math.isnan(float(pred[i]))]
        return sum(errors) / len(errors) if errors else None

    if use_lagrange:
        mae_lag = calc_mae(pred_lag)
    if use_newton_unequal or use_newton_equal:
        mae_new = calc_mae(pred_new)

    # ================== VẼ ĐỒ THỊ CHÍNH ==================
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Đồ thị nội suy</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Nếu X là số thứ tự (từ cột ngày), dùng nhãn gốc để hiện trên hover
    has_labels = "x_labels" in dir() and x_labels != x_all
    def get_label(i):
        return str(x_labels[i]) if has_labels else str(x_all[i])

    # Dữ liệu gốc
    valid_labels = [get_label(i) for i in range(len(x_all)) if not math.isnan(float(y_work[i] if y_work[i] is not None else float('nan')))]
    fig.add_trace(go.Scatter(
        x=valid_x, y=valid_y,
        mode="lines+markers", name="Dữ liệu gốc",
        text=valid_labels,
        hovertemplate="<b>%{text}</b><br>Giá trị: %{y:.4f}<extra></extra>",
        line=dict(color="#2980b9", width=1.5),
        marker=dict(size=5)
    ))

    # Điểm NaN (Ground Truth nếu có)
    gt_idx = [i for i in missing_idx if y_true[i] is not None and not math.isnan(float(y_true[i]))]
    gt_x = [x_all[i] for i in gt_idx]
    gt_y = [y_true[i] for i in gt_idx]
    if gt_x:
        fig.add_trace(go.Scatter(
            x=gt_x, y=gt_y, mode="markers",
            name="Giá trị thực (Ground Truth)",
            text=[get_label(i) for i in gt_idx],
            hovertemplate="<b>%{text}</b><br>Thực: %{y:.4f}<extra></extra>",
            marker=dict(color="orange", size=12, symbol="circle-open", line=dict(width=2))
        ))

    # Đường nội suy dày (300 điểm)
    x_dense = np.linspace(target_min, target_max, 300)

    if use_lagrange:
        y_dense_lag = []
        for xv in x_dense:
            nx, ny = (get_k_points(float(xv), valid_x, valid_y, k)
                      if mode == "Local" else (valid_x, valid_y))
            val = lagrange(nx, ny, float(xv))
            y_dense_lag.append(val)
        fig.add_trace(go.Scatter(
            x=x_dense, y=y_dense_lag, mode="lines", name="Đường Lagrange",
            line=dict(dash="dash", color="red", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[x_all[i] for i in missing_idx],
            y=[pred_lag[i] for i in missing_idx],
            mode="markers", name="Điểm nội suy Lagrange",
            text=[get_label(i) for i in missing_idx],
            hovertemplate="<b>%{text}</b><br>Lagrange: %{y:.4f}<extra></extra>",
            marker=dict(color="red", size=10, symbol="star")
        ))

    if use_newton_unequal or use_newton_equal:
        y_dense_new = []
        if mode == "Global" and use_newton_unequal:
            global_table_plot = divided_difference(valid_x, valid_y)
        for xv in x_dense:
            if mode == "Local":
                nx, ny = get_k_points(float(xv), valid_x, valid_y, k)
                if use_newton_equal and is_equally_spaced(nx):
                    val = newton_forward(nx, ny, float(xv))
                else:
                    tbl = divided_difference(nx, ny)
                    val = newton_unequal(nx, tbl, float(xv)) if tbl else None
            else:
                nx, ny = valid_x, valid_y
                if use_newton_equal and is_equally_spaced(nx):
                    val = newton_forward(nx, ny, float(xv))
                else:
                    val = newton_unequal(nx, global_table_plot, float(xv)) if global_table_plot else None
            y_dense_new.append(val)

        label_new = "Newton (cách đều)" if use_newton_equal else "Newton (không cách đều)"
        fig.add_trace(go.Scatter(
            x=x_dense, y=y_dense_new, mode="lines", name=f"Đường {label_new}",
            line=dict(dash="dot", color="#27ae60", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[x_all[i] for i in missing_idx],
            y=[pred_new[i] for i in missing_idx],
            mode="markers", name=f"Điểm nội suy {label_new}",
            text=[get_label(i) for i in missing_idx],
            hovertemplate="<b>%{text}</b><br>" + label_new + ": %{y:.4f}<extra></extra>",
            marker=dict(color="#27ae60", size=10, symbol="x")
        ))

    # Giới hạn trục Y theo dữ liệu gốc để tránh Runge kéo giãn đồ thị
    y_valid_clean = [v for v in valid_y if v is not None and not math.isnan(float(v))]
    y_margin = (max(y_valid_clean) - min(y_valid_clean)) * 0.3
    y_lo = min(y_valid_clean) - y_margin
    y_hi = max(y_valid_clean) + y_margin

    # Clip các điểm nội suy outlier ra khỏi đường dense (vẫn giữ None để đứt đường)
    def clip_dense(y_list):
        return [v if (v is not None and not math.isnan(float(v)) and y_lo <= v <= y_hi) else None
                for v in y_list]

    if use_lagrange and y_dense_lag:
        # Cập nhật lại trace đường Lagrange với giá trị đã clip
        for trace in fig.data:
            if trace.name == "Đường Lagrange":
                trace.y = clip_dense(y_dense_lag)

    if (use_newton_unequal or use_newton_equal) and y_dense_new:
        label_new_check = "Newton (cách đều)" if use_newton_equal else "Newton (không cách đều)"
        for trace in fig.data:
            if trace.name == f"Đường {label_new_check}":
                trace.y = clip_dense(y_dense_new)

    padding = max((target_max - target_min) * 0.05, (max_x_val - min_x_val) * 0.02)
    fig.update_layout(
        xaxis=dict(
            range=[target_min - padding, target_max + padding],
            title=dict(text="X", font=dict(color="#222222", size=13)),
            tickfont=dict(color="#222222", size=12),
            gridcolor="#dddddd",
            linecolor="#aaaaaa",
        ),
        yaxis=dict(
            range=[y_lo, y_hi],
            title=dict(text="Y", font=dict(color="#222222", size=13)),
            tickfont=dict(color="#222222", size=12),
            gridcolor="#dddddd",
            linecolor="#aaaaaa",
        ),
        title=dict(
            text=f"Nội suy đa thức | Chế độ: {mode_display} | k={k}"
                 + (" | Trục X: số thứ tự ngày" if has_labels else ""),
            font=dict(color="#111111", size=15),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#111111", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_color="#111111",
            bordercolor="#aaaaaa",
        ),
        height=480,
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#222222"),
    )

    # Cảnh báo nếu có outlier bị clip
    all_dense = (y_dense_lag if use_lagrange else []) + (y_dense_new if (use_newton_unequal or use_newton_equal) else [])
    n_outlier = sum(1 for v in all_dense if v is not None and not math.isnan(float(v)) and not (y_lo <= v <= y_hi))
    if n_outlier > 0:
        st.warning(f"⚠️ Phát hiện **{n_outlier} điểm** nội suy bị dao động mạnh (hiện tượng Runge) và đã bị ẩn khỏi đồ thị. Thử giảm k hoặc chuyển sang chế độ Lân cận.")

    st.plotly_chart(fig, use_container_width=True)

    # ================== KẾT QUẢ & ĐÁNH GIÁ ==================
    st.markdown('<div class="section-header">📊 Kết quả & Đánh giá</div>', unsafe_allow_html=True)

    def fmt_mae(v):
        return f"{v:.6f}" if v is not None else "N/A"

    if "So sánh" in method_raw:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔴 Lagrange")
            st.metric("⏱ Thời gian", f"{time_lag * 1000:.3f} ms")
            st.metric("📏 MAE", fmt_mae(mae_lag))
        with c2:
            st.markdown("#### 🟢 Newton (không cách đều)")
            st.metric("⏱ Thời gian", f"{time_new * 1000:.3f} ms")
            st.metric("📏 MAE", fmt_mae(mae_new))

        if mae_lag is not None and mae_new is not None:
            diff = abs(mae_lag - mae_new)
            if diff < 1e-8:
                st.info("📐 **Nhận xét:** Lagrange và Newton là hai cách xây dựng khác nhau của **cùng một đa thức nội suy duy nhất** — do đó kết quả xấp xỉ hoàn toàn trùng nhau về mặt toán học.")
            else:
                st.info(f"📐 Chênh lệch MAE giữa hai phương pháp: {diff:.2e} (do sai số làm tròn số thực).")

        if time_lag > 0 and time_new > 0:
            if time_new < time_lag:
                st.success(f"🚀 Newton nhanh hơn Lagrange **{time_lag / time_new:.1f}× lần** do tái sử dụng bảng tỷ sai phân.")
            else:
                st.info("⏱ Ở số điểm nhỏ, tốc độ hai thuật toán chưa chênh lệch đáng kể.")

    elif use_lagrange:
        st.metric("⏱ Thời gian Lagrange", f"{time_lag * 1000:.3f} ms")
        st.metric("📏 MAE Lagrange", fmt_mae(mae_lag))
    else:
        label_new = "Newton (cách đều)" if use_newton_equal else "Newton (không cách đều)"
        st.metric(f"⏱ Thời gian {label_new}", f"{time_new * 1000:.3f} ms")
        st.metric(f"📏 MAE {label_new}", fmt_mae(mae_new))

    # Đánh giá chất lượng
    mae_used = mae_lag if use_lagrange else mae_new
    if mae_used is not None:
        if mae_used < 0.5:
            st.success("✔️ **Sai số rất thấp** — nội suy đạt chất lượng tốt.")
        elif mae_used < 2.0:
            st.info("ℹ️ **Sai số trung bình** — có thể cải thiện bằng cách điều chỉnh k.")
        else:
            st.warning("⚠️ **Sai số cao** — xem xét giảm k hoặc thay đổi chế độ nội suy.")
    else:
        st.warning("📌 Không tính được MAE vì không có Ground Truth tại các điểm NaN (dữ liệu gốc bị thiếu thực sự).")

    # ================== NHẬN XÉT CHUYÊN MÔN ==================
    st.markdown('<div class="section-header">💡 Nhận xét chuyên môn</div>', unsafe_allow_html=True)

    remarks = []

    # Chế độ
    if mode == "Global":
        remarks.append(
            "🔴 **Chế độ Toàn cục (Global):** Toàn bộ dữ liệu được dùng để xây một đa thức bậc cao. "
            "Quan sát đồ thị có thể thấy **hiện tượng Runge** — đường cong dao động mạnh ở hai đầu, "
            "đặc biệt khi số điểm lớn. Đây là hạn chế kinh điển của nội suy đa thức bậc cao."
        )
    else:
        if k >= 10:
            remarks.append(
                f"⚠️ **Cảnh báo k={k}:** Bậc đa thức = {k-1} khá cao. Trong chế độ Lân cận, "
                "hiện tượng Runge cục bộ vẫn có thể xảy ra. Nên thử k ≤ 7 để so sánh."
            )
        else:
            remarks.append(
                f"✅ **Chế độ Lân cận k={k}:** Chỉ dùng {k} điểm gần nhất — đa thức bậc {k-1}. "
                "Cách tiếp cận an toàn, hạn chế hiện tượng Runge, phù hợp dữ liệu thực tế."
            )

    # Newton cách đều vs không cách đều
    if use_newton_equal:
        remarks.append(
            "📐 **Newton sai phân tiến (mốc cách đều):** Công thức dùng bảng sai phân tiến Δ. "
            "Chỉ hiệu quả khi các mốc x cách đều nhau — nếu không cách đều, chương trình tự chuyển về tỷ sai phân."
        )
    if use_newton_unequal and "So sánh" not in method_raw:
        remarks.append(
            "📐 **Newton tỷ sai phân (mốc không cách đều):** Dùng bảng tỷ sai phân chia (divided differences). "
            "Ưu điểm: tái sử dụng kết quả khi thêm mốc mới (không cần tính lại toàn bộ)."
        )

    for r in remarks:
        st.markdown(r)

    # ================== PHÂN TÍCH k ==================
    if mode == "Local" and len(valid_x) > 4:
        st.markdown('<div class="section-header">📉 Ảnh hưởng của k đến sai số MAE</div>', unsafe_allow_html=True)

        k_max = min(12, len(valid_x) - 1)
        k_values = list(range(2, k_max + 1))
        mae_list = []

        for test_k in k_values:
            temp_pred = y_work.copy()
            for i in missing_idx:
                nx, ny = get_k_points(x_all[i], valid_x, valid_y, test_k)
                val = lagrange(nx, ny, x_all[i])
                temp_pred[i] = val if val is not None else float("nan")
            err = calc_mae(temp_pred)
            mae_list.append(err if err is not None else 0)

        if any(v > 0 for v in mae_list):
            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(
                x=k_values, y=mae_list, mode="lines+markers",
                marker=dict(color="#e74c3c", size=8),
                line=dict(color="#c0392b", width=2),
                name="MAE theo k"
            ))
            best_k = k_values[mae_list.index(min(mae_list))]
            fig_k.add_vline(x=best_k, line_dash="dash", line_color="green",
                            annotation_text=f"k tốt nhất = {best_k}")
            fig_k.update_layout(
                title="Sai số MAE của Lagrange theo số điểm k (chế độ Lân cận)",
                xaxis_title="k (số điểm nội suy)",
                yaxis_title="MAE",
                height=340,
                paper_bgcolor="white", plot_bgcolor="#f8f9fa"
            )
            st.plotly_chart(fig_k, use_container_width=True)
            st.caption(f"💡 k tốt nhất trong thử nghiệm này: **k = {best_k}** (MAE = {min(mae_list):.4f})")

    # ================== BẢNG TỶ SAI PHÂN ==================
    if not use_newton_equal and len(valid_x) <= 12:
        with st.expander("🗂️ Xem bảng Tỷ Sai Phân (Divided Differences)"):
            st.markdown("**Bảng tỷ sai phân** dùng để xây dựng đa thức Newton (mốc không cách đều):")
            demo_x = valid_x[:min(6, len(valid_x))]
            demo_y = valid_y[:min(6, len(valid_y))]
            tbl = divided_difference(demo_x, demo_y)
            if tbl:
                n_show = len(demo_x)
                rows = []
                for i in range(n_show):
                    row = {"x": f"{demo_x[i]:.4f}", "f(x)": f"{demo_y[i]:.4f}"}
                    for j in range(1, n_show - i):
                        row[f"Δ^{j}"] = f"{tbl[i][j]:.4f}"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows).fillna("—"), use_container_width=True)
                st.caption("Hàng đầu tiên (i=0) của mỗi cột chính là hệ số của đa thức Newton.")

    # ================== DOWNLOAD ==================
    st.markdown("---")
    result_df = df.copy()
    if use_lagrange:
        result_df[f"Lagrange_Pred_{y_col}"] = pred_lag
    if use_newton_unequal or use_newton_equal:
        label = "Newton_Equal" if use_newton_equal else "Newton_Unequal"
        result_df[f"{label}_Pred_{y_col}"] = pred_new

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Tải xuống dữ liệu đã nội suy (CSV)",
        data=csv_bytes,
        file_name="result_interpolation.csv",
        mime="text/csv"
    )