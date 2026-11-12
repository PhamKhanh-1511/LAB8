"""Kiểm thử tự động bằng streamlit.testing.v1.AppTest (chạy: python test_app.py)."""
from streamlit.testing.v1 import AppTest

import iris_model as im

CASES = [  # (SL, SW, PL, PW, loài mong đợi)
    (5.1, 3.5, 1.4, 0.2, "setosa"),
    (6.0, 2.9, 4.5, 1.5, "versicolor"),
    (6.5, 3.0, 5.8, 2.2, "virginica"),
]


def run_case(sl, sw, pl, pw):
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception, at.exception
    for key, v in zip(("sl", "sw", "pl", "pw"), (sl, sw, pl, pw)):
        at.slider(key=key).set_value(v)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


if __name__ == "__main__":
    # 1. Trước khi bấm nút: chưa có kết quả
    at0 = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at0.exception and len(at0.slider) == 4 and len(at0.columns) >= 2
    assert not any("được dự đoán" in m.value for m in at0.markdown), "Kết quả không được hiện trước khi bấm nút"
    print("✔ Chưa bấm nút -> chưa hiện kết quả; có 4 slider, 2 cột")

    # 2. Ba ca kiểm thử bắt buộc
    for sl, sw, pl, pw, exp in CASES:
        at = run_case(sl, sw, pl, pw)
        txt = " ".join(m.value for m in at.markdown)
        assert f"**{exp.upper()}**" in txt, (sl, sw, pl, pw, txt)
        assert len(at.get("progress")) == 3
        print(f"✔ ({sl}, {sw}, {pl}, {pw}) -> {exp.upper()}")

    # 4. Cache: sau nhiều lần rerun mô hình vẫn chỉ huấn luyện 1 lần
    at = run_case(*CASES[1][:4])
    at.slider(key="sl").set_value(7.0).run()
    at.slider(key="pl").set_value(6.0).run()
    assert im.TRAIN_CALLS == 1, im.TRAIN_CALLS
    print(f"✔ Sau nhiều lần bấm nút / kéo slider, train_model() chỉ chạy {im.TRAIN_CALLS} lần (cache hoạt động)")

    # 3. Ca nằm giữa 2 loài
    sp, pr = im.predict(im.train_model(), 6.0, 2.7, 5.0, 1.7)
    print("✔ Ca giữa 2 loài (6.0, 2.7, 5.0, 1.7):", sp, {k: f"{v:.2%}" for k, v in pr.items()})

    # 5. Xử lý thiếu ảnh: không văng lỗi, hiện warning
    import os, shutil
    shutil.move("images/setosa.jpg", "images/_bak.jpg")
    try:
        at = run_case(5.1, 3.5, 1.4, 0.2)
        assert len(at.warning) == 1 and "Không tìm thấy ảnh" in at.warning[0].value
        print("✔ Thiếu file ảnh -> cảnh báo, app vẫn chạy")
    finally:
        shutil.move("images/_bak.jpg", "images/setosa.jpg")
    print("\nTẤT CẢ TEST ĐỀU ĐẠT")
