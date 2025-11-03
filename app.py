"""
FastAPI Backend - Trích xuất dữ liệu JD và CV từ Base Hiring API
"""
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime, date, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
from time import time
import pdfplumber
from io import BytesIO

app = FastAPI(
    title="Base Hiring API - JD và CV Extractor",
    description="API để trích xuất dữ liệu JD (Job Description) và CV từ Base Hiring API. Hỗ trợ AI phân tích và đánh giá ứng viên.",
    version="v1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_API_KEY = os.getenv('BASE_API_KEY', 'YOUR_FALLBACK_API_KEY_IF_NEEDED')

# Cache configuration
CACHE_TTL = 300  # 5 phút cache
_cache = {
    'openings': {'data': None, 'timestamp': 0},
    'job_descriptions': {'data': None, 'timestamp': 0}
}

# =================================================================
# Helper Functions (Không thay đổi)
# =================================================================

def get_base_openings(api_key, use_cache=True):
    """Truy xuất vị trí tuyển dụng đang hoạt động từ Base API (có cache)"""
    current_time = time()
    if not api_key:
        raise HTTPException(status_code=500, detail="BASE_API_KEY chưa được cấu hình")
    
    # Kiểm tra cache nếu được bật
    if use_cache and _cache['openings']['data'] is not None:
        if current_time - _cache['openings']['timestamp'] < CACHE_TTL:
            return _cache['openings']['data']
    
    url = "https://hiring.base.vn/publicapi/v2/opening/list"
    payload = {'access_token': api_key}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Lỗi kết nối đến Base API: {e}")

    data = response.json()
    openings = data.get('openings', [])
    
    # Lọc vị trí với trạng thái '10' (đang hoạt động)
    filtered_openings = [
        {"id": opening['id'], "name": opening['name']}
        for opening in openings
        if opening.get('status') == '10'
    ]
    
    # Lưu vào cache
    if use_cache:
        _cache['openings'] = {'data': filtered_openings, 'timestamp': current_time}
    
    return filtered_openings

def get_job_descriptions(api_key, use_cache=True):
    """Truy xuất JD (Job Description) từ các vị trí tuyển dụng đang mở (có cache)"""
    current_time = time()
    if not api_key:
        raise HTTPException(status_code=500, detail="BASE_API_KEY chưa được cấu hình")

    # Kiểm tra cache
    if use_cache and _cache['job_descriptions']['data'] is not None:
        if current_time - _cache['job_descriptions']['timestamp'] < CACHE_TTL:
            return _cache['job_descriptions']['data']
    
    url = "https://hiring.base.vn/publicapi/v2/opening/list"
    payload = {'access_token': api_key}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Lỗi kết nối đến Base API: {e}")

    data = response.json()
    
    if 'openings' in data:
        openings = data['openings']
        results = []
        
        for opening in openings:
            if opening.get('status') == '10':  # Chỉ lấy vị trí đang mở
                html_content = opening.get('content', '')
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()
                
                if len(text_content) >= 10:  # Chỉ lấy JD có nội dung đủ dài
                    results.append({
                        "id": opening['id'],
                        "name": opening['name'],
                        "job_description": text_content.strip(),
                        "html_content": html_content
                    })
        
        if use_cache:
            _cache['job_descriptions'] = {'data': results, 'timestamp': current_time}
        
        return results
    return []

def extract_message(evaluations):
    """Trích xuất nội dung văn bản từ đánh giá HTML"""
    if isinstance(evaluations, list) and len(evaluations) > 0:
        raw_html = evaluations[0].get('content', '')
        soup = BeautifulSoup(raw_html, "html.parser")
        text = " ".join(soup.stripped_strings)
        return text
    return None

def extract_text_from_pdf(url):
    """Trích xuất text từ PDF URL bằng pdfplumber"""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Trang {page_num} ---\n"
                    text += page_text
        return text.strip() if text else None
    except Exception as e:
        return None

def get_candidates_for_opening(opening_id, api_key, start_date=None, end_date=None, include_full_info=False):
    """Truy xuất ứng viên cho một vị trí tuyển dụng cụ thể trong khoảng thời gian"""
    url = "https://hiring.base.vn/publicapi/v2/candidate/list"
    
    payload = {
        'access_token': api_key,
        'opening_id': opening_id,
    }
    
    if start_date:
        payload['start_date'] = start_date.strftime('%Y-%m-%d') if isinstance(start_date, date) else start_date
    if end_date:
        payload['end_date'] = end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else end_date
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Lỗi kết nối đến Base API khi lấy ứng viên: {e}")

    data = response.json()
    if 'candidates' in data and data['candidates']:
        candidates = []
        for candidate in data['candidates']:
            cv_urls = candidate.get('cvs', [])
            cv_url = cv_urls[0] if isinstance(cv_urls, list) and len(cv_urls) > 0 else None
            
            # Tối ưu: Chỉ trích xuất text CV nếu include_full_info=True
            # Vì đây là tác vụ chậm
            cv_text = None
            if cv_url and include_full_info: 
                cv_text = extract_text_from_pdf(cv_url)
            
            review = extract_message(candidate.get('evaluations', []))
            
            form_data = {}
            if 'form' in candidate and isinstance(candidate['form'], list):
                for item in candidate['form']:
                    if isinstance(item, dict) and 'id' in item and 'value' in item:
                        form_data[item['id']] = item['value']
            
            candidate_info = {
                "id": candidate.get('id'),
                "name": candidate.get('name'),
                "email": candidate.get('email'),
                "phone": candidate.get('phone'),
                "gender": candidate.get('gender'),
                "cv_url": cv_url,
                "review": review, # Giữ lại review cơ bản
                "form_data": form_data, # Giữ lại form data cơ bản
                "opening_id": opening_id
            }
            
            if include_full_info:
                candidate_info.update({
                    "cv_text": cv_text, # cv_text chỉ có khi full_info
                    "status": candidate.get('status'),
                    "stage": candidate.get('stage'),
                    "since": candidate.get('since'),
                    "evaluations": candidate.get('evaluations', []),
                    "cvs": candidate.get('cvs', []),
                    "raw_data": candidate
                })
            
            candidates.append(candidate_info)
        
        return candidates
    return []

def find_candidate_by_id(candidate_id, api_key, opening_id=None):
    """Tìm ứng viên theo ID - tối ưu bằng cách chỉ tìm trong opening_id nếu được cung cấp"""
    if opening_id:
        candidates = get_candidates_for_opening(opening_id, api_key, None, None, include_full_info=True)
        candidate = next((c for c in candidates if c['id'] == candidate_id), None)
        if candidate:
            openings = get_base_openings(api_key, use_cache=True)
            opening = next((op for op in openings if op['id'] == opening_id), None)
            return candidate, opening
        return None, None
    else:
        openings = get_base_openings(api_key, use_cache=True)
        for opening in openings:
            candidates = get_candidates_for_opening(
                opening['id'], 
                api_key, 
                None, 
                None,
                include_full_info=True
            )
            candidate = next((c for c in candidates if c['id'] == candidate_id), None)
            if candidate:
                return candidate, opening
        return None, None

# =================================================================
# Request/Response Models (Không thay đổi)
# =================================================================

class JobDescriptionResponse(BaseModel):
    id: str
    name: str
    job_description: str

class CandidateResponse(BaseModel):
    id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    gender: Optional[str]
    cv_url: Optional[str]
    cv_text: Optional[str] # Thêm cv_text vào response model
    review: Optional[str]
    form_data: dict
    opening_id: str

# =================================================================
# API Endpoints (ĐÃ CẬP NHẬT)
# =================================================================

@app.get("/", operation_id="healthCheck")
async def root():
    """
    Health check - Kiểm tra trạng thái API và xem danh sách các endpoint.
    
    HƯỚNG DẪN CHO AI:
    - Dùng để kiểm tra xem API có hoạt động không.
    - Không dùng cho mục đích lấy dữ liệu.
    """
    return {
        "status": "ok",
        "message": "Base Hiring API - Trích xuất JD và CV",
        "endpoints": {
            "get_openings": "/api/openings",
            "get_opening_details": "/api/opening/{opening_id}",
            "get_all_candidates": "/api/candidates",
            "get_candidate_form_ai": "/api/ai/candidate-form-data/{candidate_id}",
            "get_cv_evaluation_data_ai": "/api/ai/cv-evaluation-data/{candidate_id}",
            "get_hiring_process_ai": "/api/ai/hiring-process/{opening_id}",
            "get_all_analysis_data_ai": "/api/ai/all-analysis-data"
        }
    }

@app.get("/api/openings", operation_id="layDanhSachViTriTuyenDung")
async def get_openings(
    use_cache: bool = Query(True, description="Sử dụng cache để tăng tốc độ (mặc định: true). Nên luôn dùng 'true' trừ khi muốn làm mới dữ liệu.")
):
    """
    Lấy danh sách TẤT CẢ vị trí tuyển dụng (Openings) đang hoạt động.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng hỏi "có những vị trí tuyển dụng nào?", "liệt kê các job đang mở".
    - API này chỉ trả về ID và Tên của vị trí.
    - Để lấy chi tiết (JD, ứng viên) của một vị trí, hãy dùng API `/api/opening/{opening_id}`.
    """
    try:
        openings = get_base_openings(BASE_API_KEY, use_cache=use_cache)
        return {
            "success": True,
            "total_records": len(openings),
            "openings": openings
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách vị trí: {str(e)}")


@app.get("/api/opening/{opening_id}", operation_id="layChiTietViTriVaUngVien")
async def get_opening_with_candidates(
    opening_id: str = Path(..., description="ID của vị trí tuyển dụng cần xem chi tiết."),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả.")
):
    """
    Lấy thông tin CHI TIẾT của MỘT vị trí tuyển dụng, bao gồm JD (Mô tả công việc) và danh sách ứng viên CƠ BẢN của vị trí đó.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng muốn xem JD (Mô tả công việc) CỦA MỘT VỊ TRÍ CỤ THỂ.
    - Dùng khi người dùng muốn xem danh sách ứng viên CỦA MỘT VỊ TRÍ CỤ THỂ.
    - API này KHÔNG lấy text từ CV, chỉ lấy thông tin cơ bản.
    """
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Ngày kết thúc phải sau ngày bắt đầu")
        
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            # Thử làm mới cache nếu không tìm thấy
            jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
            jd = next((jd for jd in jds if jd['id'] == opening_id), None)
            if not jd:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy JD cho opening_id: {opening_id}")
        
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        opening = next((op for op in openings if op['id'] == opening_id), None)
        opening_name = opening['name'] if opening else None
        
        # Lấy candidates (include_full_info=False: không lấy cv_text)
        candidates = get_candidates_for_opening(opening_id, BASE_API_KEY, start_date_obj, end_date_obj, include_full_info=False)
        
        return {
            "success": True,
            "opening_id": opening_id,
            "opening_name": opening_name,
            "job_description": jd['job_description'],
            "job_description_html": jd['html_content'],
            "total_candidates": len(candidates),
            "candidates": candidates
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu: {str(e)}")

@app.get("/api/candidates", operation_id="layTatCaUngVien")
async def get_all_candidates(
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    opening_id: Optional[str] = Query(None, description="ID của vị trí (tùy chọn). Nếu cung cấp, chỉ lấy ứng viên của vị trí này. Nếu BỎ TRỐNG, lấy của TẤT CẢ vị trí.")
):
    """
    Lấy danh sách ứng viên CƠ BẢN từ TẤT CẢ các vị trí, hoặc lọc theo MỘT vị trí cụ thể.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng hỏi "liệt kê ứng viên" hoặc "tìm ứng viên" nói chung.
    - Nếu `opening_id` được cung cấp, API chỉ trả về ứng viên cho vị trí đó.
    - Nếu `opening_id` BỊ BỎ TRỐNG, API sẽ trả về ứng viên từ TẤT CẢ các vị trí.
    - API này KHÔNG lấy text từ CV (cv_text sẽ là null).
    """
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Ngày kết thúc phải sau ngày bắt đầu")
        
        all_candidates = []
        
        if opening_id:
            openings = get_base_openings(BASE_API_KEY, use_cache=True)
            opening = next((op for op in openings if op['id'] == opening_id), None)
            if not opening:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí với ID: {opening_id}")
            
            candidates = get_candidates_for_opening(opening_id, BASE_API_KEY, start_date_obj, end_date_obj, include_full_info=False)
            all_candidates.extend(candidates)
        else:
            openings = get_base_openings(BASE_API_KEY, use_cache=True)
            for opening in openings:
                candidates = get_candidates_for_opening(
                    opening['id'], 
                    BASE_API_KEY, 
                    start_date_obj, 
                    end_date_obj,
                    include_full_info=False
                )
                all_candidates.extend(candidates)
        
        return {
            "success": True,
            "total_records": len(all_candidates),
            "candidates": all_candidates
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy ứng viên: {str(e)}")

# =================================================================
# AI Analysis Endpoints - Tối ưu cho AI
# =================================================================

@app.get("/api/ai/candidate-form-data/{candidate_id}", operation_id="layDuLieuFormUngVienChoAI")
async def get_candidate_form_data_for_ai(
    candidate_id: str = Path(..., description="ID của ứng viên cần lấy dữ liệu form."),
    opening_id: Optional[str] = Query(None, description="ID vị trí tuyển dụng (tùy chọn). Cung cấp để tìm ứng viên nhanh hơn.")
):
    """
    Lấy dữ liệu form ứng viên CHI TIẾT (đã được gán nhãn tiếng Việt) và NỘI DUNG CV (cv_text) của MỘT ứng viên.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng muốn PHÂN TÍCH NỘI DUNG FORM (ví dụ: điểm mạnh, điểm yếu, mong muốn...) của MỘT ứng viên cụ thể.
    - API này trả về cả `cv_text` (nội dung CV đã được bóc tách).
    - API này cũng trả về `form_data` (có nhãn tiếng Việt) và `form_data_raw` (dữ liệu gốc).
    """
    try:
        candidate_found, opening = find_candidate_by_id(candidate_id, BASE_API_KEY, opening_id)
        
        if not candidate_found:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ứng viên với ID: {candidate_id}")
        
        opening_info = {
            "opening_id": opening['id'],
            "opening_name": opening['name']
        } if opening else None
        
        # Mapping form fields (Giữ nguyên mapping của bạn)
        form_field_mapping = {
            'tinh_cach_cua_ban': 'Tính cách của bạn',
            'diem_manh_cua_ban': 'Điểm mạnh của bạn',
            'diem_yeu_cua_ban': 'Điểm yếu của bạn',
            'nhung_nguoi_xung_quanh_dong_nghi': 'Những người xung quanh đánh giá',
            'ban_rat_mong_muon_duoc_hoc_hoi_c': 'Mong muốn được học hỏi',
            'doi_voi_ban_cong_viec_nao_se_man': 'Công việc bạn thích làm',
            'doi_voi_ban_cong_viec_nao_se_lam': 'Công việc bạn không thích làm',
            'qua_trinh_hoc_tap_ghi_chu_bao_go': 'Quá trình học tập',
            'cac_bang_cap_khoa_huan_luyen_va_': 'Bằng cấp và khóa huấn luyện',
            'muc_luong_de_nghi': 'Mức lương đề nghị',
            'thoi_gian_bat_dau_lam_viec': 'Thời gian bắt đầu làm việc',
            'vui_long_cho_biet_vi_sao_ban_qua': 'Lý do ứng tuyển',
            'muc_luong_gan_nhat': 'Mức lương gần nhất',
            'ben_canh_nhung_thong_tin_trong_c': 'Thông tin bổ sung',
            'muc_luong_mong_muon': 'Mức lương mong muốn',
            'cap_bac_mong_muon': 'Cấp bậc mong muốn',
            'cong_ty_gan_day_nhat': 'Công ty gần đây nhất',
            'tinh_trang_hon_nhan': 'Tình trạng hôn nhân'
        }
        
        labeled_form_data = {}
        for key, value in candidate_found.get('form_data', {}).items():
            label = form_field_mapping.get(key, key)
            labeled_form_data[label] = {
                "field_id": key,
                "value": value
            }
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "candidate_name": candidate_found.get('name'),
            "opening_info": opening_info,
            "form_data": labeled_form_data,
            "form_data_raw": candidate_found.get('form_data', {}),
            "review": candidate_found.get('review'),
            "cv_url": candidate_found.get('cv_url'),
            "cv_text": candidate_found.get('cv_text'), # Trả về cv_text
            "status": candidate_found.get('status'),
            "stage": candidate_found.get('stage')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu form ứng viên: {str(e)}")

@app.get("/api/ai/hiring-process/{opening_id}", operation_id="layThongTinQuyTrinhTuyenDungChoAI")
async def get_hiring_process_for_ai(
    opening_id: str = Path(..., description="ID của vị trí tuyển dụng cần phân tích quy trình."),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả.")
):
    """
    Lấy thông tin THỐNG KÊ và CHI TIẾT quy trình tuyển dụng của MỘT vị trí.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng muốn PHÂN TÍCH QUY TRÌNH (ví dụ: "có bao nhiêu ứng viên ở vòng phỏng vấn?", "thống kê trạng thái ứng viên").
    - API này trả về `process_statistics` (thống kê) VÀ `candidates_detail` (danh sách ứng viên chi tiết, BAO GỒM CẢ cv_text).
    - Đây là API tốt nhất để lấy dữ liệu CHI TIẾT của TẤT CẢ ứng viên thuộc MỘT vị trí.
    """
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        opening = next((op for op in openings if op['id'] == opening_id), None)
        
        if not opening:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí với ID: {opening_id}")
        
        # Lấy candidates với include_full_info=True ĐỂ LẤY CV_TEXT
        candidates = get_candidates_for_opening(
            opening_id, 
            BASE_API_KEY, 
            start_date_obj, 
            end_date_obj,
            include_full_info=True
        )
        
        process_stats = {
            "total_candidates": len(candidates),
            "by_stage": {},
            "by_status": {},
            "interview_scheduled_not_attended": []
        }
        
        stage_counts = {}
        status_counts = {}

        for candidate in candidates:
            stage = candidate.get('stage', 'Không xác định')
            status = candidate.get('status', 'Không xác định')
            
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Logic tìm ứng viên không đến (ví dụ)
            if 'interview' in str(stage).lower() and ('no_show' in str(status).lower() or 'not_attended' in str(status).lower()):
                process_stats['interview_scheduled_not_attended'].append({
                    "candidate_id": candidate['id'],
                    "candidate_name": candidate.get('name'),
                    "stage": stage,
                    "status": status
                })
        
        process_stats['by_stage'] = stage_counts
        process_stats['by_status'] = status_counts

        return {
            "success": True,
            "opening_id": opening_id,
            "opening_name": opening['name'],
            "period": {
                "start_date": start_date_obj.isoformat() if start_date_obj else None,
                "end_date": end_date_obj.isoformat() if end_date_obj else None
            },
            "process_statistics": process_stats,
            "candidates_detail": candidates # Trả về danh sách chi tiết
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin quy trình: {str(e)}")

@app.get("/api/ai/cv-evaluation-data/{candidate_id}", operation_id="layDuLieuSoSanhJdVaCvChoAI")
async def get_cv_evaluation_data_for_ai(
    candidate_id: str = Path(..., description="ID của ứng viên cần đánh giá."),
    opening_id: Optional[str] = Query(None, description="ID vị trí tuyển dụng (tùy chọn). Cung cấp để tìm ứng viên và JD nhanh hơn.")
):
    """
    Lấy dữ liệu TỔNG HỢP (JD + CV + Form) của MỘT ứng viên để AI so sánh và đánh giá.
    
    HƯỚNG DẪN CHO AI:
    - Dùng API này khi người dùng muốn "đánh giá ứng viên", "so sánh CV với JD", "chấm điểm ứng viên".
    - API này trả về 3 phần chính:
      1. `job_description`: Mô tả công việc ứng viên đã ứng tuyển.
      2. `cv_text`: Nội dung CV đã bóc tách của ứng viên.
      3. `form_data`: Dữ liệu form ứng viên đã điền.
    """
    try:
        candidate_found, opening = find_candidate_by_id(candidate_id, BASE_API_KEY, opening_id)
        
        if not candidate_found:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ứng viên với ID: {candidate_id}")
        
        opening_info = {
            "opening_id": opening['id'],
            "opening_name": opening['name']
        } if opening else None
        
        jd_text = None
        jd_html = None
        
        if opening_info:
            jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
            jd = next((jd for jd in jds if jd['id'] == opening_info['opening_id']), None)
            if jd:
                jd_text = jd['job_description']
                jd_html = jd['html_content']
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "candidate_name": candidate_found.get('name'),
            "opening_info": opening_info,
            "job_description": jd_text,
            "job_description_html": jd_html,
            "cv_url": candidate_found.get('cv_url'),
            "cv_text": candidate_found.get('cv_text'), # Đã bao gồm trong find_candidate_by_id
            "form_data": candidate_found.get('form_data', {}),
            "review": candidate_found.get('review')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu đánh giá CV: {str(e)}")

@app.get("/api/ai/all-analysis-data", operation_id="layTatCaDuLieuPhanTichChoAI")
async def get_all_analysis_data_for_ai(
    opening_id: Optional[str] = Query(None, description="ID vị trí (tùy chọn). Nếu cung cấp, chỉ lấy dữ liệu của vị trí này. Nếu BỎ TRỐNG, lấy TẤT CẢ."),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc (YYYY-MM-DD). Bỏ trống để lấy tất cả.")
):
    """
    Lấy TẤT CẢ dữ liệu (Tất cả JD, Tất cả ứng viên CHI TIẾT) để AI phân tích tổng thể.
    
    HƯỚNG DẪN CHO AI:
    - Đây là API "nặng" nhất. Chỉ dùng khi người dùng muốn PHÂN TÍCH TỔNG THỂ, so sánh giữa các vị trí, hoặc khi các API khác không đủ thông tin.
    - Có thể lọc theo MỘT `opening_id` hoặc lấy TẤT CẢ (nếu `opening_id` bỏ trống).
    - API này trả về `job_descriptions` (danh sách JD) và `candidates` (danh sách ứng viên CHI TIẾT, BAO GỒM CẢ cv_text).
    """
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        openings_to_fetch = get_base_openings(BASE_API_KEY, use_cache=True)
        
        if opening_id:
            openings_to_fetch = [op for op in openings_to_fetch if op['id'] == opening_id]
            jds = [jd for jd in jds if jd['id'] == opening_id]
        
        all_candidates = []
        for opening in openings_to_fetch:
            # Lấy candidates với include_full_info=True ĐỂ LẤY CV_TEXT
            candidates = get_candidates_for_opening(
                opening['id'],
                BASE_API_KEY,
                start_date_obj,
                end_date_obj,
                include_full_info=True
            )
            all_candidates.extend(candidates)
        
        return {
            "success": True,
            "total_openings": len(openings_to_fetch),
            "total_candidates": len(all_candidates),
            "period": {
                "start_date": start_date_obj.isoformat() if start_date_obj else None,
                "end_date": end_date_obj.isoformat() if end_date_obj else None
            },
            "job_descriptions": jds,
            "candidates": all_candidates
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu phân tích: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
