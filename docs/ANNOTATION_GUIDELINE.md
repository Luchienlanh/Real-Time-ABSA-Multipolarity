#  Hướng Dẫn Đánh Nhãn (Annotation Guideline)

## **Real-Time Streaming Sentiment Analysis - Target-Oriented E-commerce**

---

##  Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Định Nghĩa Các Khía Cạnh](#2-định-nghĩa-các-khía-cạnh-aspects)
3. [Hệ Thống Nhãn Cảm Xúc](#3-hệ-thống-nhãn-cảm-xúc)
4. [Quy Trình Đánh Nhãn](#4-quy-trình-đánh-nhãn)
5. [Các Trường Hợp Đặc Biệt - Đa Cực](#5-các-trường-hợp-đặc-biệt---đa-cực-multi-polarity)
6. [Ví Dụ Cụ Thể](#6-ví-dụ-cụ-thể)
7. [Các Lỗi Thường Gặp](#7-các-lỗi-thường-gặp)
8. [Checklist Trước Khi Nộp](#8-checklist-trước-khi-nộp)

---

## 1. Giới Thiệu

### 1.1 Mục Đích

Tài liệu này hướng dẫn người đánh nhãn (annotator) cách gắn nhãn cảm xúc cho các bình luận e-commerce theo phương pháp **Aspect-Based Sentiment Analysis (ABSA)** - Phân tích cảm xúc theo từng khía cạnh.

### 1.2 Nguyên Tắc Cốt Lõi

- **Danh từ** trong câu bình luận → xác định **khía cạnh** được nhắc đến
- **Tính từ/trạng từ bổ sung** → xác định **cảm xúc** cho khía cạnh đó
- Mỗi khía cạnh có thể có **nhiều cực cảm xúc** (đa cực/multi-polarity)

### 1.3 Nguồn Dữ Liệu

Dữ liệu được lấy từ thư mục: `data/test_flow/`

---

## 2. Định Nghĩa Các Khía Cạnh (Aspects)

Hệ thống sử dụng **9 khía cạnh** được tối ưu hóa cho thương mại điện tử:

| #   | Khía Cạnh                   | Ký Hiệu | Mô Tả                                  | Từ Khóa Liên Quan                                       |
| --- | --------------------------- | ------- | -------------------------------------- | ------------------------------------------------------- |
| 1   | **Chất lượng sản phẩm**     | CL      | Chất liệu, độ bền, hoàn thiện sản phẩm | vải, chất, form, bền, nặng, nhẹ, mịn, thô, cứng, mềm    |
| 2   | **Hiệu năng & Trải nghiệm** | HN      | Trải nghiệm sử dụng, hiệu suất         | dùng, xài, sử dụng, hoạt động, chạy, pin, nhanh, chậm   |
| 3   | **Đúng mô tả**              | MT      | Độ chính xác so với mô tả/hình ảnh     | giống hình, đúng mô tả, như ảnh, khác hình, không giống |
| 4   | **Giá cả & Khuyến mãi**     | GC      | Giá tiền, ưu đãi, giá trị              | giá, tiền, rẻ, đắt, hời, voucher, mã giảm, sale         |
| 5   | **Vận chuyển**              | VC      | Tốc độ, chất lượng giao hàng           | ship, giao, nhanh, chậm, shipper, đơn vị vận chuyển     |
| 6   | **Đóng gói**                | DG      | Bao bì, đóng gói sản phẩm              | đóng gói, hộp, bọc, cẩn thận, bubble, bị móp, bẹp       |
| 7   | **Dịch vụ & Thái độ Shop**  | DV      | CSKH, thái độ người bán                | shop, seller, hỗ trợ, tư vấn, nhiệt tình, trả lời       |
| 8   | **Bảo hành & Đổi trả**      | BH      | Chính sách bảo hành, đổi trả           | bảo hành, đổi, trả, hoàn, lỗi, sửa chữa                 |
| 9   | **Tính xác thực**           | XT      | Hàng thật/giả, nguồn gốc               | chính hãng, auth, fake, nhái, real, thật, giả           |

---

## 3. Hệ Thống Nhãn Cảm Xúc

### 3.1 Bảng Giá Trị Nhãn

| Giá Trị | Ý Nghĩa                           | Mô Tả Chi Tiết                                       |
| ------- | --------------------------------- | ---------------------------------------------------- |
| **1**   |  Tích cực (Positive)            | Khách hàng hài lòng, khen ngợi về khía cạnh này      |
| **0**   |  Trung lập (Neutral)            | Nhắc đến nhưng không thể hiện cảm xúc rõ ràng        |
| **-1**  |  Tiêu cực (Negative)            | Khách hàng không hài lòng, phàn nàn về khía cạnh này |
| **2**   |  Không nhắc đến (Not Mentioned) | Bình luận không đề cập đến khía cạnh này             |

### 3.2 Định Nghĩa Từng Loại Nhãn

####  Tích Cực (1)

Sử dụng khi người bình luận thể hiện sự **hài lòng, khen ngợi, đánh giá cao**:

- Từ khóa: "tốt", "đẹp", "ưng", "thích", "ok", "ổn", "nhanh", "chất lượng", "xuất sắc"...
- Biểu cảm: icon vui, lời cảm ơn, đề nghị mua lại

####  Trung Lập (0)

Sử dụng khi người bình luận **nhắc đến nhưng không bày tỏ cảm xúc rõ ràng**:

- Mô tả khách quan mà không đánh giá
- Thông tin trung tính (ví dụ: "nhận được áo màu đen")
- Câu hỏi hoặc thắc mắc

####  Tiêu Cực (-1)

Sử dụng khi người bình luận thể hiện sự **không hài lòng, phàn nàn, chê**:

- Từ khóa: "tệ", "xấu", "dở", "chậm", "lỗi", "hỏng", "thất vọng", "không như"...
- Biểu cảm: icon buồn/giận, đề nghị hoàn tiền, cảnh báo người khác

####  Không Nhắc Đến (2)

Sử dụng khi bình luận **hoàn toàn không đề cập** đến khía cạnh đó:

- Không có từ khóa liên quan
- Không có ngữ cảnh gián tiếp

---

## 4. Quy Trình Đánh Nhãn

### Bước 1: Đọc Kỹ Bình Luận

- Đọc toàn bộ bình luận ít nhất 2 lần
- Xác định ngữ cảnh tổng thể (khen/chê/trung lập)

### Bước 2: Nhận Diện Danh Từ → Khía Cạnh

- Tìm các **danh từ** trong câu
- Map danh từ về **khía cạnh tương ứng** (xem bảng Section 2)

### Bước 3: Nhận Diện Tính Từ → Cảm Xúc

- Tìm các **tính từ/trạng từ** bổ sung cho danh từ
- Xác định **cảm xúc** dựa trên:
  - Từ mang nghĩa tích cực/tiêu cực
  - Ngữ cảnh câu
  - Biểu cảm emoji (nếu có)

### Bước 4: Gán Nhãn

- Điền giá trị nhãn vào cột khía cạnh tương ứng
- Mặc định: **2** (không nhắc đến) cho các khía cạnh không liên quan

### Bước 5: Kiểm Tra Đa Cực

- Xem xét có khía cạnh nào **vừa được khen VÀ chê** không
- Nếu có → sử dụng định dạng đa cực (xem Section 5)

---

## 5. Các Trường Hợp Đặc Biệt - Đa Cực (Multi-Polarity)

> ️ **QUAN TRỌNG**: Đây là tính năng đặc biệt của hệ thống!

### 5.1 Khi Nào Sử Dụng Đa Cực?

Khi một khía cạnh trong **cùng một bình luận** nhận được **nhiều hơn một loại cảm xúc**:

- Vừa khen vừa chê
- Có cả ý kiến tích cực và tiêu cực
- Phần này tốt, phần khác không tốt

### 5.2 Định Dạng Đá Cực

| Trường Hợp                 | Cách Ghi                   | Ý Nghĩa                           |
| -------------------------- | -------------------------- | --------------------------------- |
| Vừa tích cực vừa tiêu cực  | `[-1, 1]` hoặc `1,-1`      | Mixed: có cả khen và chê          |
| Vừa tích cực vừa trung lập | `[0, 1]` hoặc `0,1`        | Có khen nhưng cũng có ý trung lập |
| Vừa tiêu cực vừa trung lập | `[-1, 0]` hoặc `-1,0`      | Có chê nhưng cũng có ý trung lập  |
| Tất cả 3 loại              | `[-1, 0, 1]` hoặc `-1,0,1` | Phức tạp: có cả 3 ý kiến          |

### 5.3 Ví Dụ Đa Cực

**Ví dụ 1:** _"Áo đẹp nhưng vải hơi mỏng"_

- Khía cạnh **Chất lượng sản phẩm**:
  - "đẹp" → Tích cực (+1)
  - "vải mỏng" → Tiêu cực (-1)
  - → Đánh: **[-1, 1]**

**Ví dụ 2:** _"Giá rẻ so với chất lượng nhưng ship hơi chậm, giao hàng cẩn thận"_

- **Giá cả & Khuyến mãi**: "rẻ" → **1**
- **Vận chuyển**: "chậm" → **-1**
- **Đóng gói**: "cẩn thận" → **1**

**Ví dụ 3:** _"Sản phẩm chất lượng tốt cho giá tiền này, form hơi rộng một chút"_

- **Chất lượng sản phẩm**:
  - "chất lượng tốt" → Tích cực (+1)
  - "form hơi rộng" → Tiêu cực (-1)
  - → Đánh: **[-1, 1]**
- **Giá cả & Khuyến mãi**: "cho giá tiền này" → **1**

---

## 6. Ví Dụ Cụ Thể

### 6.1 Ví Dụ Đơn Giản

| Bình Luận                          | CL  | HN  | MT  | GC  | VC  | DG  | DV  | BH  | XT  |
| ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| "Giày đẹp, ship nhanh"             | 1   | 2   | 2   | 2   | 1   | 2   | 2   | 2   | 2   |
| "Hàng giả, thất vọng"              | 2   | 2   | 2   | 2   | 2   | 2   | 2   | 2   | -1  |
| "Giao hàng chậm, đóng gói cẩu thả" | 2   | 2   | 2   | 2   | -1  | -1  | 2   | 2   | 2   |
| "Tốt"                              | 1   | 2   | 2   | 2   | 2   | 2   | 2   | 2   | 2   |

### 6.2 Ví Dụ Đa Cực

| Bình Luận                                  | CL     | HN  | MT  | GC  | VC  | DG  | DV  | BH  | XT  |
| ------------------------------------------ | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| "Áo đẹp nhưng vải hơi mỏng"                | [-1,1] | 2   | 2   | 2   | 2   | 2   | 2   | 2   | 2   |
| "Giá rẻ, chất ổn nhưng size hơi lớn"       | [-1,1] | 2   | 2   | 1   | 2   | 2   | 2   | 2   | 2   |
| "Ship nhanh nhưng hộp bị móp, sản phẩm ok" | 1      | 2   | 2   | 2   | 1   | -1  | 2   | 2   | 2   |

### 6.3 Ví Dụ Trung Lập

| Bình Luận           | Giải Thích                                   |
| ------------------- | -------------------------------------------- |
| "Nhận được rồi"     | Không đánh giá → **2** cho tất cả            |
| "Áo size M màu đen" | Mô tả thông tin thuần túy → **2** cho tất cả |
| "Đang chờ dùng thử" | Chưa có đánh giá → **2** cho tất cả          |

---

## 7. Các Lỗi Thường Gặp

###  Lỗi 1: Nhầm Lẫn Giữa Các Khía Cạnh

| Sai                            | Đúng                          |
| ------------------------------ | ----------------------------- |
| "Ship nhanh" → Chất lượng SP   | "Ship nhanh" → **Vận chuyển** |
| "Đóng hộp đẹp" → Chất lượng SP | "Đóng hộp đẹp" → **Đóng gói** |

###  Lỗi 2: Quên Đa Cực

| Bình Luận           | Sai    | Đúng             |
| ------------------- | ------ | ---------------- |
| "Áo đẹp nhưng mỏng" | CL = 1 | CL = **[-1, 1]** |

###  Lỗi 3: Đánh Cảm Xúc Theo Cảm Nhận Riêng

- **Đúng**: Dựa vào từ ngữ trong bình luận
- **Sai**: Đoán ý người viết

###  Lỗi 4: Bỏ Sót Khía Cạnh Ngầm

| Bình Luận                                | Khía Cạnh Ngầm              |
| ---------------------------------------- | --------------------------- |
| "Giao hàng siêu nhanh, cảm ơn a shipper" | Vận chuyển = 1, Dịch vụ = 1 |
| "Được giá này thì chất ok"               | Giá cả = 1, Chất lượng = 1  |

---

## 8. Checklist Trước Khi Nộp

Trước khi hoàn thành, hãy kiểm tra:

- [ ] Đã đọc tất cả bình luận trong file
- [ ] Mỗi bình luận đều có nhãn cho TẤT CẢ 9 khía cạnh
- [ ] Các khía cạnh không nhắc đến đều được đánh **2**
- [ ] Đã kiểm tra và ghi nhận các trường hợp **đa cực**
- [ ] Không để trống ô nào
- [ ] Đã review lại các bình luận dài/phức tạp

---

##  Liên Hệ Hỗ Trợ

Nếu gặp trường hợp khó hoặc không chắc chắn:

1. Ghi chú lại ID/vị trí của bình luận
2. Đánh nhãn theo đánh giá tốt nhất của bạn
3. Báo cáo khi nộp file để được review lại

---

##  Tóm Tắt Nhanh

```
┌─────────────────────────────────────────────────────────────┐
│                    BẢNG GIÁ TRỊ NHÃN                        │
├─────────┬───────────────────────────────────────────────────┤
│   1     │   Tích cực (Positive)                           │
│   0     │   Trung lập (Neutral)                           │
│  -1     │   Tiêu cực (Negative)                           │
│   2     │   Không nhắc đến (Not Mentioned)                │
│ [-1,1]  │   Đa cực: Vừa tích cực vừa tiêu cực             │
└─────────┴───────────────────────────────────────────────────┘
```

---

**Phiên bản:** 1.0  
**Ngày tạo:** 24/12/2024  
**Dự án:** Real-Time Streaming Sentiment Analysis - Target-Oriented E-commerce
