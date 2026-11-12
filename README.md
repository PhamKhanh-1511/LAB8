# Demo phân loại hoa Iris bằng Streamlit (Gaussian Naive Bayes)

**Họ tên:** Phạm Khanh – **MSSV:** SE204991 – **Môn:** AIL303m

Viết lại demo Gradio thành Streamlit, giữ nguyên chức năng và bố cục 2 cột (trái: 4 slider + nút *Dự đoán ngay*; phải: tên loài, ảnh, xác suất 3 lớp).

## Cấu trúc
| File | Vai trò |
|---|---|
| `iris_model.py` | Huấn luyện GaussianNB (toàn bộ Iris) + dự đoán + đánh giá Test 80/20 |
| `app.py` | Giao diện Streamlit, `@st.cache_resource` cho mô hình |
| `make_images.py` | Tạo 3 ảnh minh họa vào `images/` (có thể thay bằng ảnh thật cùng tên) |
| `test_app.py` | Kiểm thử tự động bằng `streamlit.testing.v1.AppTest` |
| `SE204991_PhamKhanh_Streamlit.ipynb` | Notebook nộp bài: ghi file, kiểm thử, chạy local/Colab |

## Chạy trên máy cá nhân
```bash
pip install -r requirements.txt
python make_images.py        # chỉ cần nếu chưa có thư mục images/
streamlit run app.py         # mở http://localhost:8501
python test_app.py           # kiểm thử tự động
```
Trên Google Colab: mở notebook `SE204991_PhamKhanh_Streamlit.ipynb` và chạy lần lượt các ô (ô mục 5 mở tunnel `localtunnel`).

## Báo cáo kiểm thử
| STT | Đầu vào (SL, SW, PL, PW) | Mong đợi | Thực tế | Setosa / Versicolor / Virginica |
|---|---|---|---|---|
| 1 | 5.1, 3.5, 1.4, 0.2 | Setosa | **SETOSA** ✔ | 100.00% / 0.00% / 0.00% |
| 2 | 6.0, 2.9, 4.5, 1.5 | Versicolor | **VERSICOLOR** ✔ | 0.00% / 98.65% / 1.35% |
| 3 | 6.5, 3.0, 5.8, 2.2 | Virginica | **VIRGINICA** ✔ | 0.00% / 0.00% / 100.00% |
| 4 | 6.0, 2.7, 5.0, 1.7 | giữa 2 loài | **VIRGINICA** | 0.00% / 40.99% / 59.01% |

Ảnh chụp màn hình: [chưa bấm nút](screenshots/test0_truoc_khi_bam.png) · [Test 1](screenshots/test1_setosa.png) · [Test 2](screenshots/test2_versicolor.png) · [Test 3](screenshots/test3_virginica.png) · [Test 4](screenshots/test4_giua_2_loai.png)

![Test 1](screenshots/test1_setosa.png)
![Test 2](screenshots/test2_versicolor.png)
![Test 3](screenshots/test3_virginica.png)
![Test 4](screenshots/test4_giua_2_loai.png)

**Nhận xét ca 4:** mẫu (PL = 5.0, PW = 1.7) nằm ở vùng chồng lấn giữa Versicolor và Virginica. Mô hình chọn Virginica nhưng chỉ 59,01%, còn Versicolor tới 40,99%; Setosa gần như 0% vì đã tách hẳn khỏi hai loài còn lại. Chênh lệch giữa hai loài gần nhau chưa tới 20 điểm phần trăm nên dự đoán kém chắc chắn, trong khi ba ca còn lại đều ≥ 98,65%. Điều này khớp với accuracy 96,67% trên tập Test: các mẫu bị nhầm đều nằm giữa Versicolor và Virginica.

## Bonus đã làm
- Sidebar: thông tin mô hình, accuracy Test 80/20 (`random_state=42`, `stratify=y`) = 96,67%, bộ đếm số lần huấn luyện (luôn = 1 nhờ cache).
- Tab bảng dữ liệu Iris (`st.dataframe`) và tab scatter plot đánh dấu ⭐ điểm vừa nhập.
- Thiếu ảnh minh họa → `st.warning`, ứng dụng không bị dừng.

## Ghi chú
- Ảnh trong `images/` là hình vẽ minh họa cách điệu bằng code, không phải ảnh chụp thật.
- Kết quả chỉ hiện sau khi bấm nút và được giữ trong `st.session_state` cho tới lần bấm tiếp theo (giống hành vi của bản Gradio).
