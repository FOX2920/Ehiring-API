# Base Hiring API - Job Description & CV Extractor

API để trích xuất và quản lý dữ liệu tuyển dụng từ Base Hiring platform, bao gồm Job Descriptions (JD), thông tin ứng viên, lịch phỏng vấn và offer letters.

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [API Endpoints](#-api-endpoints)
- [Sơ đồ hoạt động](#-sơ-đồ-hoạt-động)
- [Ví dụ sử dụng](#-ví-dụ-sử-dụng)

## 🚀 Tính năng chính

- ✅ Trích xuất Job Description từ các vị trí tuyển dụng
- ✅ Lấy danh sách và chi tiết ứng viên
- ✅ Tự động trích xuất text từ CV (PDF/DOCX) bằng Gemini AI
- ✅ Quản lý lịch phỏng vấn
- ✅ Trích xuất Offer Letter
- ✅ Tìm kiếm thông minh với Cosine Similarity
- ✅ Cache dữ liệu để tối ưu hiệu suất
- ✅ Tích hợp Google Sheet cho dữ liệu bài test

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- pip

### Cài đặt dependencies

```bash
pip install fastapi uvicorn requests beautifulsoup4 pdfplumber scikit-learn numpy python-dotenv google-generativeai pytz python-docx
```

### Chạy server

```bash
python base_hiring_api.py
```

Server sẽ chạy tại: `http://localhost:8000`

## ⚙️ Cấu hình

Tạo file `.env` trong thư mục gốc:

```env
# Bắt buộc
BASE_API_KEY=your_base_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Tùy chọn
GEMINI_API_KEY_DU_PHONG=key1,key2,key3  # API keys dự phòng (phân cách bằng dấu phẩy)
GOOGLE_SHEET_SCRIPT_URL=your_google_sheet_script_url
ACCOUNT_API_KEY=your_account_api_key
```

### Giải thích các biến môi trường

| Biến | Mô tả | Bắt buộc |
|------|-------|----------|
| `BASE_API_KEY` | API key của Base Hiring | ✅ |
| `GEMINI_API_KEY` | API key của Google Gemini (trích xuất CV) | ✅ |
| `GEMINI_API_KEY_DU_PHONG` | API keys dự phòng khi chính bị rate limit | ❌ |
| `GOOGLE_SHEET_SCRIPT_URL` | URL script để lấy dữ liệu bài test | ❌ |
| `ACCOUNT_API_KEY` | API key để lấy thông tin users cho reviews | ❌ |

## 🔌 API Endpoints

### 1. Health Check
```http
GET /
```

Kiểm tra trạng thái API và xem danh sách endpoints.

---

### 2. Lấy Job Description

```http
GET /api/opening/job-description?opening_name_or_id={name_or_id}
```

**Tham số:**
- `opening_name_or_id` (optional): Tên hoặc ID của vị trí tuyển dụng
  - Nếu bỏ trống: Trả về tất cả openings có status 10
  - Nếu có giá trị: Tìm kiếm bằng Cosine Similarity

**Response:**
```json
{
  "success": true,
  "query": "Backend Developer",
  "opening_id": "123",
  "opening_name": "Backend Developer",
  "similarity_score": 0.95,
  "job_description": "Chi tiết JD..."
}
```

---

### 3. Lấy danh sách ứng viên

```http
GET /api/opening/{opening_name_or_id}/candidates?start_date=2024-01-01&end_date=2024-12-31&stage_name=Interviewed
```

**Tham số:**
- `opening_name_or_id` (required): Tên hoặc ID vị trí tuyển dụng
- `start_date` (optional): Ngày bắt đầu lọc (YYYY-MM-DD)
- `end_date` (optional): Ngày kết thúc lọc (YYYY-MM-DD)
- `stage_name` (optional): Lọc theo stage (VD: "Interviewed", "Offered")

**Response:**
```json
{
  "success": true,
  "opening_id": "123",
  "opening_name": "Backend Developer",
  "job_description": "...",
  "total_candidates": 10,
  "candidates": [
    {
      "id": "candidate_123",
      "name": "Nguyễn Văn A",
      "email": "email@example.com",
      "cv_text": "Extracted CV content...",
      "reviews": [
        {
          "name": "Hoang Tran",
          "title": "CEO",
          "content": "Excellent candidate..."
        }
      ],
      "test_results": [...]
    }
  ]
}
```

---

### 4. Lấy lịch phỏng vấn

```http
GET /api/interviews?opening_name_or_id=Backend&date=2024-11-15
```

**Tham số:**
- `opening_name_or_id` (optional): Lọc theo vị trí tuyển dụng
- `date` (optional): Lọc theo ngày cụ thể (YYYY-MM-DD)
- `start_date` (optional): Ngày bắt đầu
- `end_date` (optional): Ngày kết thúc

**Response:**
```json
{
  "success": true,
  "total_interviews": 5,
  "interviews": [
    {
      "id": "interview_123",
      "candidate_name": "Nguyễn Văn A",
      "opening_name": "Backend Developer",
      "time_dt": "2024-11-15T14:00:00+07:00"
    }
  ]
}
```

---

### 5. Lấy chi tiết ứng viên

```http
GET /api/candidate?candidate_id=123
```

hoặc

```http
GET /api/candidate?opening_name_or_id=Backend&candidate_name=Nguyen Van A
```

**Tham số (chọn 1 trong 2 cách):**

**Cách 1:** Tìm trực tiếp bằng ID
- `candidate_id`: ID của ứng viên

**Cách 2:** Tìm bằng tên (sử dụng Cosine Similarity)
- `opening_name_or_id`: Tên/ID vị trí tuyển dụng
- `candidate_name`: Tên ứng viên

**Response:**
```json
{
  "success": true,
  "candidate_id": "123",
  "candidate_details": {
    "id": "123",
    "ten": "Nguyễn Văn A",
    "email": "email@example.com",
    "vi_tri_ung_tuyen": "Backend Developer",
    "cv_text": "Extracted CV...",
    "job_description": "JD content...",
    "reviews": [...],
    "test_results": [...]
  }
}
```

---

### 6. Lấy Offer Letter

```http
GET /api/offer-letter?candidate_id=123
```

hoặc

```http
GET /api/offer-letter?opening_name_or_id=Backend&candidate_name=Nguyen Van A
```

**Tham số:** Giống như endpoint `/api/candidate`

**Lưu ý:** Endpoint này chỉ tìm kiếm trong các ứng viên có stage là "Offered" hoặc "Hired"

**Response:**
```json
{
  "success": true,
  "candidate_name": "Nguyễn Văn A",
  "vi_tri_ung_tuyen": "Backend Developer",
  "offer_letter": {
    "url": "https://...",
    "name": "offer_letter.pdf",
    "text": "Extracted offer letter content..."
  }
}
```

## 📊 Sơ đồ hoạt động

### 1. Luồng lấy Job Description

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/opening/job-description?opening_name_or_id=Backend
       ▼
┌─────────────────────────────────────────────────┐
│         Kiểm tra Cache (5 phút TTL)             │
├─────────────────────────────────────────────────┤
│  ✓ Có cache → Trả về ngay                       │
│  ✗ Không cache → Tiếp tục                       │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Gọi Base API: /opening/list (status=10)        │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Tìm kiếm Opening bằng Cosine Similarity        │
├─────────────────────────────────────────────────┤
│  • Exact match → similarity = 1.0               │
│  • Fuzzy match → TF-IDF vectorization           │
│  • Threshold: 0.5                               │
└──────┬──────────────────────────────────────────┘
       │
       ├─ ✓ Tìm thấy
       │  └─► Trích xuất JD từ HTML content
       │      └─► Trả về JD + similarity score
       │
       └─ ✗ Không tìm thấy
          └─► Trả về danh sách tất cả openings
```

### 2. Luồng lấy ứng viên

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/opening/{name}/candidates
       ▼
┌─────────────────────────────────────────────────┐
│  Tìm Opening ID (Cosine Similarity)             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Gọi Base API: /candidate/list                  │
│  (với opening_id, start_date, end_date)         │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Lọc theo stage_name (nếu có)                   │
│  Sử dụng Cosine Similarity để match stage       │
└──────┬──────────────────────────────────────────┘
       │
       ▼ (Song song xử lý từng ứng viên)
┌─────────────────────────────────────────────────┐
│  Với mỗi ứng viên:                              │
├─────────────────────────────────────────────────┤
│  1. Trích xuất CV text                          │
│     ├─ PDF → pdfplumber                         │
│     ├─ Fallback → Gemini AI (với retry)         │
│     └─ Rate limit → Chuyển sang API dự phòng    │
│                                                 │
│  2. Xử lý Reviews                               │
│     ├─ Lấy username từ evaluations              │
│     ├─ Map sang name + title từ Account API     │
│     └─ CEO special handling                     │
│                                                 │
│  3. Lấy Test Results từ Google Sheet            │
│                                                 │
│  4. Parse Form Data                             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Trả về danh sách ứng viên + JD                 │
└─────────────────────────────────────────────────┘
```

### 3. Luồng lấy lịch phỏng vấn

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/interviews?date=2024-11-15
       ▼
┌─────────────────────────────────────────────────┐
│  Gọi Base API: /interview/list                  │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Lọc theo Opening (nếu có opening_name_or_id)   │
│  Sử dụng Cosine Similarity                      │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Chuyển đổi timestamp → datetime                │
│  Timezone: Asia/Ho_Chi_Minh (UTC+7)             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Lọc theo date (nếu có tham số date)            │
│  So sánh date của time_dt                       │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Trả về danh sách interviews                    │
└─────────────────────────────────────────────────┘
```

### 4. Luồng lấy chi tiết ứng viên

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/candidate?opening_name_or_id=Backend&candidate_name=Nguyen Van A
       ▼
┌─────────────────────────────────────────────────┐
│  Kiểm tra tham số đầu vào                       │
├─────────────────────────────────────────────────┤
│  Cách 1: candidate_id                           │
│  Cách 2: opening_name_or_id + candidate_name    │
└──────┬──────────────────────────────────────────┘
       │
       ├─ Cách 1: candidate_id
       │  └─► Sử dụng trực tiếp
       │
       └─ Cách 2: Tìm kiếm bằng tên
          │
          ▼
       ┌─────────────────────────────────────────────────┐
       │  Tìm Opening (Cosine Similarity)                │
       └──────┬──────────────────────────────────────────┘
              │
              ▼
       ┌─────────────────────────────────────────────────┐
       │  Gọi Base API: /candidate/list (theo opening)   │
       └──────┬──────────────────────────────────────────┘
              │
              ▼
       ┌─────────────────────────────────────────────────┐
       │  Tìm Candidate bằng tên (Cosine Similarity)     │
       │  • Không lọc stage (tìm trong tất cả stages)    │
       │  • Threshold: 0.5                               │
       └──────┬──────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Gọi Base API: /candidate/get                   │
│  Lấy dữ liệu chi tiết (raw response)            │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Làm phẳng dữ liệu (flatten)                    │
│  • fields → key-value pairs                     │
│  • form → key-value pairs                       │
│  • evaluations → reviews với name + title       │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Trích xuất CV text (Gemini AI)                 │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Lấy Test Results (Google Sheet)                │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Tìm và thêm Job Description                    │
│  (dựa trên opening_name từ evaluations)         │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Trả về candidate_details (JSON phẳng)          │
└─────────────────────────────────────────────────┘
```

### 5. Luồng lấy Offer Letter

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/offer-letter?opening_name_or_id=Backend&candidate_name=Nguyen Van A
       ▼
┌─────────────────────────────────────────────────┐
│  Tìm Candidate ID (giống /api/candidate)        │
│  • Tìm Opening (Cosine Similarity)              │
│  • Tìm Candidate (Cosine Similarity)            │
│  ⚠️  CHỈ TÌM trong stage: Offered, Hired        │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Gọi Base API: /candidate/messages              │
│  Lấy tất cả tin nhắn của ứng viên               │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Duyệt tin nhắn từ mới → cũ                     │
│  Tìm file PDF/DOCX/DOC                          │
├─────────────────────────────────────────────────┤
│  Ưu tiên:                                       │
│  1. Attachments (has_attachment > 0)            │
│  2. Links trong HTML content                    │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Tải file đầu tiên tìm được                     │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Trích xuất text                                │
├─────────────────────────────────────────────────┤
│  • PDF → pdfplumber                             │
│  • DOCX → python-docx                           │
│  • DOC → Không hỗ trợ                           │
└──────┬──────────────────────────────────────────┘
       │
       ├─ ✓ Trích xuất thành công
       │  └─► Trả về offer letter + candidate info
       │
       └─ ✗ Không tìm thấy hoặc lỗi
          └─► HTTP 404
```

## 💡 Ví dụ sử dụng

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Lấy Job Description
response = requests.get(f"{BASE_URL}/api/opening/job-description", params={
    "opening_name_or_id": "Backend Developer"
})
jd_data = response.json()
print(f"JD: {jd_data['job_description']}")

# 2. Lấy ứng viên
response = requests.get(f"{BASE_URL}/api/opening/Backend Developer/candidates", params={
    "start_date": "2024-01-01",
    "stage_name": "Interviewed"
})
candidates = response.json()
print(f"Tổng số ứng viên: {candidates['total_candidates']}")

# 3. Lấy chi tiết ứng viên
response = requests.get(f"{BASE_URL}/api/candidate", params={
    "opening_name_or_id": "Backend Developer",
    "candidate_name": "Nguyen Van A"
})
details = response.json()
print(f"Email: {details['candidate_details']['email']}")

# 4. Lấy offer letter
response = requests.get(f"{BASE_URL}/api/offer-letter", params={
    "candidate_id": "123"
})
offer = response.json()
print(f"Offer letter: {offer['offer_letter']['text']}")
```

### cURL

```bash
# Lấy Job Description
curl "http://localhost:8000/api/opening/job-description?opening_name_or_id=Backend"

# Lấy ứng viên với lọc ngày
curl "http://localhost:8000/api/opening/Backend%20Developer/candidates?start_date=2024-01-01&end_date=2024-12-31"

# Lấy lịch phỏng vấn ngày hôm nay
curl "http://localhost:8000/api/interviews?date=2024-11-15"

# Lấy chi tiết ứng viên
curl "http://localhost:8000/api/candidate?opening_name_or_id=Backend&candidate_name=Nguyen%20Van%20A"
```

## 🔍 Tính năng đặc biệt

### 1. Cosine Similarity Search

API sử dụng TF-IDF và Cosine Similarity để tìm kiếm thông minh:

- **Opening name**: "Backend Dev" → tìm được "Backend Developer"
- **Candidate name**: "Nguyen A" → tìm được "Nguyễn Văn A"
- **Stage name**: "Interview" → tìm được "Interviewed"

**Threshold mặc định**: 0.5 (có thể điều chỉnh)

### 2. Cache System

- **TTL**: 5 phút
- **Cached data**:
  - Danh sách openings
  - Job descriptions
  - Users info (cho reviews)

### 3. Gemini AI Fallback

Khi trích xuất CV:
1. Thử `pdfplumber` trước (nhanh, miễn phí)
2. Nếu thất bại → Gemini AI chính
3. Nếu rate limit → Chuyển sang API dự phòng

### 4. Review Processing

- Tự động map username → tên thật + chức danh
- Special handling: "Hoang Tran" → CEO
- Format: `[Name - Title] Content`

## 🐛 Xử lý lỗi

| HTTP Code | Ý nghĩa |
|-----------|---------|
| 200 | Thành công |
| 400 | Tham số không hợp lệ |
| 404 | Không tìm thấy dữ liệu |
| 500 | Lỗi server |
| 503 | Không kết nối được Base API |

## 📝 Lưu ý

1. **Rate Limiting**: Gemini API có giới hạn request. Sử dụng API keys dự phòng.
2. **Cache**: Dữ liệu được cache 5 phút. Force refresh bằng cách restart server.
3. **Timezone**: Tất cả datetime được convert sang Asia/Ho_Chi_Minh (UTC+7).
4. **File Support**: 
   - ✅ PDF (pdfplumber + Gemini)
   - ✅ DOCX (python-docx)
   - ❌ DOC (không hỗ trợ)

## 🔐 Bảo mật

- ⚠️ Không commit file `.env` lên Git
- ⚠️ API keys phải được bảo mật
- ✅ CORS được bật cho development (nên hạn chế trong production)

## 📚 Tài liệu tham khảo

- [Base Hiring API Documentation](https://hiring.base.vn/publicapi)
- [Google Gemini API](https://ai.google.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

MIT License

---
