"""Phần HUẤN LUYỆN + DỰ ĐOÁN (không phụ thuộc Streamlit, có thể test độc lập).

- load_dataframe(): tải Iris dạng DataFrame (để xem bảng + vẽ biểu đồ)
- train_model()   : huấn luyện GaussianNB trên TOÀN BỘ dữ liệu Iris (giống bản Gradio)
- evaluate_holdout(): độ chính xác trên tập Test (chia 80/20, random_state=42, stratify=y)
- predict()       : dự đoán loài + xác suất 3 lớp cho một mẫu hoa
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Đếm số lần train_model() thực sự chạy -> dùng để chứng minh @st.cache_resource hoạt động
# (module chỉ được import 1 lần nên biến này không bị reset khi Streamlit chạy lại script)
TRAIN_CALLS = 0


def load_iris_data():
    """Trả về (X, y, tên_cột_đặc_trưng, tên_loài) của tập Iris."""
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names), [str(n) for n in iris.target_names]


def load_dataframe() -> pd.DataFrame:
    """Trả về DataFrame Iris gồm 4 đặc trưng + cột 'species' (tên loài)."""
    X, y, feature_names, target_names = load_iris_data()
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]
    return df


def train_model() -> GaussianNB:
    """Huấn luyện Gaussian Naive Bayes trên toàn bộ 150 mẫu Iris và trả về mô hình."""
    global TRAIN_CALLS
    TRAIN_CALLS += 1
    X, y, _, _ = load_iris_data()
    model = GaussianNB()
    model.fit(X, y)
    return model


def evaluate_holdout() -> float:
    """Độ chính xác trên tập Test: chia 80/20, random_state=42, stratify=y (tham chiếu ≈ 96.67%)."""
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    m = GaussianNB().fit(X_train, y_train)
    return float(accuracy_score(y_test, m.predict(X_test)))


def predict(model: GaussianNB, sepal_length, sepal_width, petal_length, petal_width):
    """Dự đoán loài hoa cho một mẫu.

    Input : model đã huấn luyện + 4 kích thước (cm).
    Output: (tên loài dạng chữ thường, dict {tên loài: xác suất}).
    """
    _, _, _, target_names = load_iris_data()
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_idx = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    return target_names[pred_idx], {target_names[i]: float(proba[i]) for i in range(len(target_names))}
