# Base Hiring API - JD và CV Extractor

FastAPI Backend để trích xuất dữ liệu JD (Job Description) và CV từ Base Hiring API.

## Tính năng

- Trích xuất Job Description từ các vị trí tuyển dụng đang mở
- Lấy danh sách ứng viên theo vị trí tuyển dụng
- Trích xuất nội dung CV từ URL (ưu tiên pdfplumber, fallback về Gemini AI)
- Lấy lịch phỏng vấn theo vị trí và ngày
- Lấy chi tiết ứng viên với đầy đủ thông tin
- Tự động tìm kiếm vị trí tuyển dụng bằng cosine similarity
- Cache dữ liệu để tối ưu hiệu suất

## Yêu cầu

- Python 3.8+
- Các thư viện Python (xem requirements.txt hoặc phần Dependencies)

## Cài đặt

1. Clone repository hoặc tải file `base_hiring_api.py`

2. Cài đặt các dependencies:
```bash
pip install fastapi uvicorn requests beautifulsoup4 pdfplumber scikit-learn numpy google-genai pytz python-dotenv
```

3. Tạo file `.env` trong thư mục gốc với nội dung:
```
BASE_API_KEY=your_base_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY_DU_PHONG=backup_key1,backup_key2
```

## Cấu hình

### Biến môi trường

- `BASE_API_KEY` (bắt buộc): API key để truy cập Base Hiring API
- `GEMINI_API_KEY` (bắt buộc): API key chính cho Google Gemini AI
- `GEMINI_API_KEY_DU_PHONG` (tùy chọn): Danh sách API keys dự phòng, phân cách bằng dấu phẩy

## Chạy ứng dụng

```bash
python base_hiring_api.py
```

Hoặc sử dụng uvicorn trực tiếp:
```bash
uvicorn base_hiring_api:app --reload --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: `http://localhost:8000`

## API Endpoints

### 1. Health Check

**GET** `/`

Kiểm tra trạng thái API và xem danh sách các endpoints có sẵn.

**Response:**
```json
{
  "status": "ok",
  "message": "Base Hiring API - Trích xuất JD và CV",
  "endpoints": {
    "get_candidates": "/api/opening/{opening_name_or_id}/candidates",
    "get_job_description": "/api/opening/{opening_name_or_id}/job-description",
    "get_interviews": "/api/interviews",
    "get_candidate_details": "/api/candidate/{candidate_id}"
  }
}
```

### 2. Lấy Job Description

**GET** `/api/opening/{opening_name_or_id}/job-description`

Lấy JD (Job Description) theo tên hoặc ID của vị trí tuyển dụng. Hệ thống tự động tìm kiếm vị trí gần nhất bằng cosine similarity nếu dùng tên.

**Parameters:**
- `opening_name_or_id` (path): Tên hoặc ID của vị trí tuyển dụng

**Response:**
```json
{
  "success": true,
  "query": "opening_name_or_id",
  "opening_id": "123",
  "opening_name": "Tên vị trí chính xác",
  "similarity_score": 0.95,
  "job_description": "Nội dung JD..."
}
```

### 3. Lấy danh sách ứng viên

**GET** `/api/opening/{opening_name_or_id}/candidates`

Lấy tất cả ứng viên theo vị trí tuyển dụng, bao gồm cv_text đã được trích xuất.

**Parameters:**
- `opening_name_or_id` (path): Tên hoặc ID của vị trí tuyển dụng
- `start_date` (query, tùy chọn): Ngày bắt đầu lọc (YYYY-MM-DD)
- `end_date` (query, tùy chọn): Ngày kết thúc lọc (YYYY-MM-DD)
- `stage_name` (query, tùy chọn): Lọc theo stage name

**Response:**
```json
{
  "success": true,
  "query": "opening_name_or_id",
  "opening_id": "123",
  "opening_name": "Tên vị trí",
  "similarity_score": 0.95,
  "job_description": "Nội dung JD...",
  "total_candidates": 10,
  "candidates": [
    {
      "id": "candidate_id",
      "name": "Tên ứng viên",
      "email": "email@example.com",
      "phone": "0123456789",
      "gender": "M",
      "cv_url": "https://...",
      "cv_text": "Nội dung CV đã trích xuất...",
      "review": "Đánh giá...",
      "form_data": {},
      "opening_id": "123",
      "stage_id": "456",
      "stage_name": "Đã phỏng vấn"
    }
  ]
}
```

### 4. Lấy lịch phỏng vấn

**GET** `/api/interviews`

Lấy lịch phỏng vấn, có thể lọc theo vị trí tuyển dụng và ngày.

**Parameters:**
- `opening_name_or_id` (query, tùy chọn): Tên hoặc ID của vị trí tuyển dụng
- `date` (query, tùy chọn): Lấy lịch cho 1 ngày cụ thể (YYYY-MM-DD)
- `start_date` (query, tùy chọn): Ngày bắt đầu lọc (YYYY-MM-DD)
- `end_date` (query, tùy chọn): Ngày kết thúc lọc (YYYY-MM-DD)

**Response:**
```json
{
  "success": true,
  "query": "opening_name_or_id",
  "date": "2024-01-15",
  "opening_id": "123",
  "opening_name": "Tên vị trí",
  "similarity_score": 0.95,
  "total_interviews": 5,
  "interviews": [
    {
      "id": "interview_id",
      "candidate_id": "candidate_id",
      "candidate_name": "Tên ứng viên",
      "opening_name": "Tên vị trí",
      "time_dt": "2024-01-15T10:00:00+07:00"
    }
  ]
}
```

### 5. Lấy chi tiết ứng viên

**GET** `/api/candidate/{candidate_id}`

Lấy chi tiết đầy đủ của một ứng viên, bao gồm cv_text và job_description.

**Parameters:**
- `candidate_id` (path): ID của ứng viên

**Response:**
```json
{
  "success": true,
  "candidate_id": "candidate_id",
  "candidate_details": {
    "id": "candidate_id",
    "ten": "Tên ứng viên",
    "email": "email@example.com",
    "so_dien_thoai": "0123456789",
    "vi_tri_ung_tuyen": "Tên vị trí",
    "opening_id": "123",
    "stage_id": "456",
    "stage_name": "Đã phỏng vấn",
    "cv_url": "https://...",
    "cv_text": "Nội dung CV đã trích xuất...",
    "job_description": "Nội dung JD..."
  }
}
```

## Trích xuất CV

Hệ thống sử dụng phương pháp hai bước để trích xuất nội dung CV:

1. **Ưu tiên pdfplumber**: Trích xuất text trực tiếp từ PDF URL
2. **Fallback Gemini AI**: Nếu pdfplumber không thành công, sử dụng Google Gemini AI để đọc và trích xuất text từ URL

Điều này giúp:
- Giảm số lượng request đến Gemini API
- Tăng tốc độ xử lý cho các file PDF hợp lệ
- Vẫn có phương án dự phòng cho các trường hợp đặc biệt

## Cache

Hệ thống sử dụng cache trong bộ nhớ với TTL 5 phút cho:
- Danh sách vị trí tuyển dụng (openings)
- Danh sách Job Descriptions

Cache giúp giảm số lượng request đến Base API và cải thiện hiệu suất.

## Cosine Similarity

Hệ thống sử dụng cosine similarity với TF-IDF vectorization để:
- Tìm kiếm vị trí tuyển dụng gần nhất khi người dùng nhập tên không chính xác
- Tìm kiếm stage name phù hợp khi lọc ứng viên

## Xử lý lỗi

- **400 Bad Request**: Dữ liệu đầu vào không hợp lệ
- **404 Not Found**: Không tìm thấy tài nguyên (vị trí, ứng viên, etc.)
- **500 Internal Server Error**: Lỗi server
- **503 Service Unavailable**: Lỗi kết nối đến Base API hoặc Gemini API

## Dependencies

- `fastapi`: Framework web API
- `uvicorn`: ASGI server
- `requests`: HTTP client
- `beautifulsoup4`: Parse HTML
- `pdfplumber`: Trích xuất text từ PDF
- `scikit-learn`: Cosine similarity và TF-IDF
- `numpy`: Xử lý mảng
- `google-genai`: Google Gemini AI client
- `pytz`: Xử lý timezone
- `python-dotenv`: Đọc biến môi trường từ .env

## Ghi chú

- Tất cả thời gian được chuyển đổi sang timezone Asia/Ho_Chi_Minh
- Hệ thống tự động retry với các API keys dự phòng khi gặp lỗi rate limit (429)
- CV text được trích xuất tự động cho tất cả ứng viên trong danh sách

## License

MIT License

