"""Phần GIAO DIỆN Streamlit – Demo phân loại hoa Iris (Gaussian Naive Bayes).

Chạy:  streamlit run app.py
Lưu ý: Streamlit chạy lại TOÀN BỘ file này mỗi khi người dùng tương tác với widget,
nên mô hình được huấn luyện trong hàm có @st.cache_resource (chỉ huấn luyện 1 lần).
"""
import os

import matplotlib.pyplot as plt
import streamlit as st

import iris_model as im

# ---------------------------------------------------------------- 1. Cấu hình trang
st.set_page_config(page_title="Iris Classification - Naive Bayes", page_icon="🌸", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ảnh minh họa của từng loài (đặt trong thư mục images/)
SPECIES_IMAGES = {
    "setosa": os.path.join(BASE_DIR, "images", "setosa.jpg"),
    "versicolor": os.path.join(BASE_DIR, "images", "versicolor.jpg"),
    "virginica": os.path.join(BASE_DIR, "images", "virginica.jpg"),
}

# Cấu hình 4 thanh trượt: (khóa, nhãn, min, max, mặc định, bước)
SLIDERS = [
    ("sl", "Chiều dài lá đài (Sepal Length - cm)", 4.0, 8.0, 5.1, 0.1),
    ("sw", "Chiều rộng lá đài (Sepal Width - cm)", 2.0, 4.5, 3.5, 0.1),
    ("pl", "Chiều dài cánh hoa (Petal Length - cm)", 1.0, 7.0, 1.4, 0.1),
    ("pw", "Chiều rộng cánh hoa (Petal Width - cm)", 0.1, 2.5, 0.2, 0.1),
]
SPECIES_COLORS = {"setosa": "#1f77b4", "versicolor": "#2ca02c", "virginica": "#d62728"}


# ---------------------------------------------------------------- 2. Cache mô hình / dữ liệu
@st.cache_resource(show_spinner="Đang huấn luyện mô hình Gaussian Naive Bayes...")
def get_model():
    """Huấn luyện GaussianNB đúng 1 lần; các lần rerun sau lấy lại từ cache."""
    return im.train_model()


@st.cache_data
def get_dataframe():
    """Bảng dữ liệu Iris (cache để không đọc lại)."""
    return im.load_dataframe()


@st.cache_data
def get_test_accuracy():
    """Độ chính xác trên tập Test 80/20 (cache vì không đổi)."""
    return im.evaluate_holdout()


# ---------------------------------------------------------------- 3. Các hàm hiển thị
def show_species_image(species: str):
    """Hiển thị ảnh minh họa; nếu thiếu file thì cảnh báo thay vì làm app bị dừng."""
    path = SPECIES_IMAGES[species]
    if os.path.exists(path):
        st.image(path, caption=f"Iris {species}", width=320)
    else:
        st.warning(f"Không tìm thấy ảnh minh họa: images/{os.path.basename(path)}")


def show_probabilities(probs: dict):
    """Hiển thị xác suất của cả 3 loài bằng thanh tiến trình."""
    for name, p in probs.items():
        st.progress(p, text=f"{name.capitalize()}: {p * 100:.2f}%")


def scatter_with_user_point(df, x_col, y_col, user_values):
    """Scatter plot phân bố 3 loài và đánh dấu điểm người dùng vừa nhập (nếu có)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for species, color in SPECIES_COLORS.items():
        sub = df[df["species"] == species]
        ax.scatter(sub[x_col], sub[y_col], c=color, label=species, alpha=0.7, s=28)
    if user_values is not None:
        names = list(df.columns[:4])
        ax.scatter(user_values[names.index(x_col)], user_values[names.index(y_col)],
                   marker="*", s=380, c="gold", edgecolors="black", linewidths=1.2,
                   label="Mẫu của bạn", zorder=5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- 4. Giao diện chính
model = get_model()
df_iris = get_dataframe()

# Sidebar (bonus): thông tin mô hình
with st.sidebar:
    st.header("ℹ️ Thông tin mô hình")
    st.markdown(
        "- **Thuật toán:** Gaussian Naive Bayes\n"
        "- **Dữ liệu:** Iris (150 mẫu, 4 đặc trưng, 3 loài)\n"
        "- **Huấn luyện:** toàn bộ 150 mẫu"
    )
    st.metric("Accuracy trên tập Test (80/20)", f"{get_test_accuracy() * 100:.2f}%")
    st.caption("Chia train/test 80/20, random_state=42, stratify=y")
    st.metric("Số lần huấn luyện mô hình", im.TRAIN_CALLS,
              help="Luôn bằng 1 nhờ @st.cache_resource, dù kéo slider bao nhiêu lần.")

# Tiêu đề + dòng phụ + hướng dẫn
st.title("Demo Phân Loại Hoa Iris Dataset")
st.subheader("Mô hình: Gaussian Naive Bayes")
st.markdown("Nhập thông tin kích thước ở cột bên trái, bấm **Dự đoán ngay** để xem loài hoa tương ứng ở cột bên phải.")

col_left, col_right = st.columns(2)

# ---- Cột trái: đầu vào
with col_left:
    st.markdown("### Nhập thông tin mẫu hoa")
    values = [st.slider(label, mn, mx, default, step, key=key)
              for key, label, mn, mx, default, step in SLIDERS]
    clicked = st.button("🔍 Dự đoán ngay", type="primary")

# Khi bấm nút: dự đoán và lưu kết quả vào session_state
# (giống Gradio: kết quả giữ nguyên cho tới lần bấm nút tiếp theo, không đổi khi chỉ kéo slider)
if clicked:
    species, probs = im.predict(model, *values)
    st.session_state["result"] = {"species": species, "probs": probs, "values": values}

result = st.session_state.get("result")

# ---- Cột phải: kết quả (chỉ hiển thị sau khi đã bấm nút)
with col_right:
    st.markdown("### Kết quả dự đoán")
    if result is None:
        st.info("Chưa có kết quả. Hãy chọn kích thước và bấm **Dự đoán ngay**.")
    else:
        st.markdown(f"### Loài hoa được dự đoán: **{result['species'].upper()}**")
        show_species_image(result["species"])
        st.markdown("**Xác suất tương ứng từng loài**")
        show_probabilities(result["probs"])
        v = result["values"]
        st.caption(f"Đầu vào đã dùng: SL={v[0]}, SW={v[1]}, PL={v[2]}, PW={v[3]} (cm)")

# ---- Bonus: bảng dữ liệu + biểu đồ phân bố
st.divider()
tab_chart, tab_data = st.tabs(["📈 Biểu đồ phân bố", "📋 Bảng dữ liệu Iris"])
with tab_chart:
    feature_cols = list(df_iris.columns[:4])
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("Trục X", feature_cols, index=2)
    y_col = c2.selectbox("Trục Y", feature_cols, index=3)
    c_plot, c_note = st.columns([3, 2])  # biểu đồ ở cột rộng, ghi chú ở cột hẹp để chữ không bị phóng to
    with c_plot:
        scatter_with_user_point(df_iris, x_col, y_col, result["values"] if result else None)
    with c_note:
        if result is None:
            st.caption("Bấm **Dự đoán ngay** để đánh dấu ⭐ mẫu của bạn trên biểu đồ.")
        else:
            st.markdown("⭐ là mẫu hoa bạn vừa nhập. Các loài càng tách xa nhau trên biểu đồ "
                        "thì mô hình càng dễ phân loại; điểm nằm giữa hai đám mây là vùng dễ nhầm.")
with tab_data:
    st.dataframe(df_iris)
