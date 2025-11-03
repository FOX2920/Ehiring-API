"""
FastAPI Backend - Trích xuất dữ liệu JD và CV từ Base Hiring API
"""
from fastapi import FastAPI, Query, HTTPException
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

app = FastAPI(title="Base Hiring API - JD và CV Extractor")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_API_KEY = os.getenv('BASE_API_KEY')

# Cache configuration
CACHE_TTL = 300  # 5 phút cache
_cache = {
    'openings': {'data': None, 'timestamp': 0},
    'job_descriptions': {'data': None, 'timestamp': 0}
}

# =================================================================
# Helper Functions
# =================================================================

def get_base_openings(api_key, use_cache=True):
    """Truy xuất vị trí tuyển dụng đang hoạt động từ Base API (có cache)"""
    current_time = time()
    
    # Kiểm tra cache nếu được bật
    if use_cache and _cache['openings']['data'] is not None:
        if current_time - _cache['openings']['timestamp'] < CACHE_TTL:
            return _cache['openings']['data']
    
    url = "https://hiring.base.vn/publicapi/v2/opening/list"
    
    payload = {'access_token': api_key}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
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
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Lỗi API: {response.text}")

def get_job_descriptions(api_key, use_cache=True):
    """Truy xuất JD (Job Description) từ các vị trí tuyển dụng đang mở (có cache)"""
    current_time = time()
    
    # Kiểm tra cache nếu được bật
    if use_cache and _cache['job_descriptions']['data'] is not None:
        if current_time - _cache['job_descriptions']['timestamp'] < CACHE_TTL:
            return _cache['job_descriptions']['data']
    
    url = "https://hiring.base.vn/publicapi/v2/opening/list"
    payload = {'access_token': api_key}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Lỗi API: {response.text}")
    
    data = response.json()
    
    if 'openings' in data:
        openings = data['openings']
        results = []
        
        for opening in openings:
            if opening.get('status') == '10':  # Chỉ lấy vị trí đang mở
                html_content = opening.get('content', '')
                # Convert HTML content to plain text
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()
                
                if len(text_content) >= 10:  # Chỉ lấy JD có nội dung đủ dài
                    results.append({
                        "id": opening['id'],
                        "name": opening['name'],
                        "job_description": text_content.strip(),
                        "html_content": html_content
                    })
        
        # Lưu vào cache
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
        # Tải file PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Đọc PDF từ bytes
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
        # Trả về None nếu lỗi, không throw exception để không làm gián đoạn quá trình
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
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        data = response.json()
        if 'candidates' in data and data['candidates']:
            candidates = []
            for candidate in data['candidates']:
                # Trích xuất CV URL
                cv_urls = candidate.get('cvs', [])
                cv_url = cv_urls[0] if isinstance(cv_urls, list) and len(cv_urls) > 0 else None
                
                # Trích xuất text từ PDF CV
                cv_text = None
                if cv_url:
                    cv_text = extract_text_from_pdf(cv_url)
                
                # Trích xuất đánh giá
                review = extract_message(candidate.get('evaluations', []))
                
                # Xử lý form data
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
                    "cv_text": cv_text,
                    "review": review,
                    "form_data": form_data,
                    "opening_id": opening_id
                }
                
                # Thêm thông tin đầy đủ nếu cần cho AI phân tích
                if include_full_info:
                    candidate_info.update({
                        "status": candidate.get('status'),
                        "stage": candidate.get('stage'),
                        "since": candidate.get('since'),  # Thời gian ứng tuyển
                        "evaluations": candidate.get('evaluations', []),  # Tất cả đánh giá
                        "cvs": candidate.get('cvs', []),  # Tất cả CV
                        "raw_data": candidate  # Dữ liệu gốc để phân tích
                    })
                
                candidates.append(candidate_info)
            
            return candidates
        return []
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Lỗi API: {response.text}")

def find_candidate_by_id(candidate_id, api_key, opening_id=None):
    """Tìm ứng viên theo ID - tối ưu bằng cách chỉ tìm trong opening_id nếu được cung cấp"""
    if opening_id:
        # Nếu biết opening_id, chỉ tìm trong đó (nhanh nhất)
        candidates = get_candidates_for_opening(opening_id, api_key, None, None, include_full_info=True)
        candidate = next((c for c in candidates if c['id'] == candidate_id), None)
        if candidate:
            openings = get_base_openings(api_key, use_cache=True)
            opening = next((op for op in openings if op['id'] == opening_id), None)
            return candidate, opening
        return None, None
    else:
        # Nếu không biết opening_id, phải duyệt qua tất cả (chậm hơn)
        # Sử dụng cache để giảm số lần gọi API
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
# Request/Response Models
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
    cv_text: Optional[str]
    review: Optional[str]
    form_data: dict
    opening_id: str

class CandidatesResponse(BaseModel):
    success: bool
    total_records: int
    opening_id: str
    opening_name: Optional[str]
    candidates: List[CandidateResponse]

# =================================================================
# API Endpoints
# =================================================================

@app.get("/")
async def root():
    """Health check - Danh sách các endpoint"""
    return {
        "status": "ok",
        "message": "Base Hiring API - Trích xuất JD và CV",
        "endpoints": {
            "opening_full": "/api/opening/{opening_id}",
            "openings": "/api/openings",
            "ai_candidate_form": "/api/ai/candidate-form-data/{candidate_id}",
            "ai_cv_evaluation": "/api/ai/cv-evaluation-data/{candidate_id}",
            "ai_all_data": "/api/ai/all-analysis-data"
        }
    }

@app.get("/api/opening/{opening_id}")
async def get_opening_with_candidates(
    opening_id: str,
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)")
):
    """
    Lấy JD và danh sách ứng viên của một vị trí tuyển dụng trong một response
    
    Trả về:
    - Thông tin vị trí (opening_id, opening_name)
    - JD đầy đủ (job_description, job_description_html)
    - Danh sách ứng viên và CV
    
    Parameters:
    - opening_id: ID của vị trí tuyển dụng
    - start_date: Ngày bắt đầu (tùy chọn, format: YYYY-MM-DD)
    - end_date: Ngày kết thúc (tùy chọn, format: YYYY-MM-DD)
    """
    try:
        # Parse dates
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Validate dates
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Ngày kết thúc phải sau ngày bắt đầu")
        
        # Lấy JD từ cache
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy JD cho opening_id: {opening_id}")
        
        # Lấy opening name từ cache
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        opening = next((op for op in openings if op['id'] == opening_id), None)
        opening_name = opening['name'] if opening else None
        
        # Lấy candidates
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

@app.get("/api/openings")
async def get_openings(use_cache: bool = Query(True, description="Sử dụng cache (mặc định: true)")):
    """
    Lấy danh sách tất cả vị trí tuyển dụng đang hoạt động
    
    Parameters:
    - use_cache: Sử dụng cache để tăng tốc độ (mặc định: true)
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

@app.get("/api/candidates")
async def get_all_candidates(
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)"),
    opening_id: Optional[str] = Query(None, description="ID của vị trí tuyển dụng (tùy chọn)")
):
    """
    Lấy danh sách ứng viên từ tất cả hoặc một vị trí tuyển dụng cụ thể
    
    Parameters:
    - start_date: Ngày bắt đầu (tùy chọn, format: YYYY-MM-DD)
    - end_date: Ngày kết thúc (tùy chọn, format: YYYY-MM-DD)
    - opening_id: ID của vị trí tuyển dụng (tùy chọn, nếu không có thì lấy tất cả)
    """
    try:
        # Parse dates
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Validate dates
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Ngày kết thúc phải sau ngày bắt đầu")
        
        all_candidates = []
        
        if opening_id:
            # Lấy ứng viên từ một vị trí cụ thể (sử dụng cache)
            openings = get_base_openings(BASE_API_KEY, use_cache=True)
            opening = next((op for op in openings if op['id'] == opening_id), None)
            
            if not opening:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí với ID: {opening_id}")
            
            candidates = get_candidates_for_opening(opening_id, BASE_API_KEY, start_date_obj, end_date_obj, include_full_info=False)
            all_candidates.extend(candidates)
        else:
            # Lấy ứng viên từ tất cả vị trí (sử dụng cache)
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
# AI Analysis Endpoints - Trích xuất dữ liệu cho AI phân tích
# =================================================================

@app.get("/api/ai/candidate-form-data/{candidate_id}")
async def get_candidate_form_data_for_ai(
    candidate_id: str,
    opening_id: Optional[str] = Query(None, description="ID vị trí tuyển dụng (tùy chọn, giúp tìm nhanh hơn)")
):
    """
    Lấy dữ liệu form ứng viên chi tiết để AI đánh giá năng lực và phù hợp
    
    Trả về:
    - Tất cả thông tin form ứng viên đã điền
    - Đánh giá hiện có (nếu có)
    - Thông tin ứng viên cơ bản
    
    Parameters:
    - candidate_id: ID của ứng viên
    - opening_id: ID vị trí tuyển dụng (tùy chọn, nếu có sẽ tìm nhanh hơn)
    """
    try:
        # Tìm ứng viên bằng helper function tối ưu
        candidate_found, opening = find_candidate_by_id(candidate_id, BASE_API_KEY, opening_id)
        
        if not candidate_found:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ứng viên với ID: {candidate_id}")
        
        opening_info = {
            "opening_id": opening['id'],
            "opening_name": opening['name']
        } if opening else None
        
        # Mapping form fields để dễ đọc
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
        
        # Tạo form data có nhãn
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
            "cv_text": candidate_found.get('cv_text'),
            "status": candidate_found.get('status'),
            "stage": candidate_found.get('stage')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu form ứng viên: {str(e)}")

@app.get("/api/ai/hiring-process/{opening_id}")
async def get_hiring_process_for_ai(
    opening_id: str,
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)")
):
    """
    Lấy thông tin quy trình tuyển dụng để AI phân tích và đề xuất cải thiện
    
    Trả về:
    - Trạng thái ứng viên ở các giai đoạn
    - Thống kê quy trình (số lượng ứng viên ở mỗi stage)
    - Thông tin về ứng viên đã hẹn phỏng vấn nhưng không đến
    - Timeline của quy trình
    """
    try:
        # Parse dates
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get opening info (sử dụng cache)
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        opening = next((op for op in openings if op['id'] == opening_id), None)
        
        if not opening:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí với ID: {opening_id}")
        
        # Get candidates with full info
        candidates = get_candidates_for_opening(
            opening_id, 
            BASE_API_KEY, 
            start_date_obj, 
            end_date_obj,
            include_full_info=True
        )
        
        # Phân tích quy trình
        process_stats = {
            "total_candidates": len(candidates),
            "by_stage": {},
            "by_status": {},
            "interview_scheduled_not_attended": [],
            "stages": []
        }
        
        for candidate in candidates:
            stage = candidate.get('stage', 'unknown')
            status = candidate.get('status', 'unknown')
            
            # Thống kê theo stage
            if stage not in process_stats['by_stage']:
                process_stats['by_stage'][stage] = []
            process_stats['by_stage'][stage].append({
                "candidate_id": candidate['id'],
                "candidate_name": candidate.get('name'),
                "status": status
            })
            
            # Thống kê theo status
            if status not in process_stats['by_status']:
                process_stats['by_status'][status] = 0
            process_stats['by_status'][status] += 1
            
            # Tìm ứng viên đã hẹn nhưng không đến (cần logic phức tạp hơn từ API)
            # Ở đây giả định status hoặc stage có thể chỉ ra điều này
            if 'interview' in str(stage).lower() or 'interview' in str(status).lower():
                if 'no_show' in str(status).lower() or 'not_attended' in str(status).lower():
                    process_stats['interview_scheduled_not_attended'].append({
                        "candidate_id": candidate['id'],
                        "candidate_name": candidate.get('name'),
                        "stage": stage,
                        "status": status
                    })
        
        return {
            "success": True,
            "opening_id": opening_id,
            "opening_name": opening['name'],
            "period": {
                "start_date": start_date_obj.isoformat() if start_date_obj else None,
                "end_date": end_date_obj.isoformat() if end_date_obj else None
            },
            "process_statistics": process_stats,
            "candidates_detail": candidates
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin quy trình: {str(e)}")

@app.get("/api/ai/cv-evaluation-data/{candidate_id}")
async def get_cv_evaluation_data_for_ai(
    candidate_id: str,
    opening_id: Optional[str] = Query(None, description="ID vị trí tuyển dụng (tùy chọn, giúp tìm nhanh hơn)")
):
    """
    Lấy dữ liệu đầy đủ để AI chấm điểm CV (JD + CV + Form data)
    
    Trả về:
    - JD của vị trí ứng tuyển
    - CV URL của ứng viên
    - Form data của ứng viên
    - Thông tin cơ bản
    
    Parameters:
    - candidate_id: ID của ứng viên
    - opening_id: ID vị trí tuyển dụng (tùy chọn, nếu có sẽ tìm nhanh hơn)
    """
    try:
        # Tìm ứng viên bằng helper function tối ưu
        candidate_found, opening = find_candidate_by_id(candidate_id, BASE_API_KEY, opening_id)
        
        if not candidate_found:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ứng viên với ID: {candidate_id}")
        
        opening_info = {
            "opening_id": opening['id'],
            "opening_name": opening['name']
        } if opening else None
        
        # Lấy JD từ cache
        if opening_info:
            jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
            jd = next((jd for jd in jds if jd['id'] == opening_info['opening_id']), None)
        else:
            jd = None
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "candidate_name": candidate_found.get('name'),
            "opening_info": opening_info,
            "job_description": jd['job_description'] if jd else None,
            "job_description_html": jd['html_content'] if jd else None,
            "cv_url": candidate_found.get('cv_url'),
            "cv_text": candidate_found.get('cv_text'),
            "form_data": candidate_found.get('form_data', {}),
            "review": candidate_found.get('review')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu đánh giá CV: {str(e)}")

@app.get("/api/ai/all-analysis-data")
async def get_all_analysis_data_for_ai(
    opening_id: Optional[str] = Query(None, description="ID vị trí (tùy chọn)"),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc (YYYY-MM-DD)")
):
    """
    Lấy tất cả dữ liệu cần thiết cho AI phân tích tổng thể
    
    Trả về:
    - Danh sách JD và yêu cầu công việc
    - Tất cả ứng viên với form data đầy đủ
    - Thống kê quy trình tuyển dụng
    """
    try:
        # Parse dates
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Lấy JD và openings (sử dụng cache)
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        
        # Lọc theo opening_id nếu có
        if opening_id:
            openings = [op for op in openings if op['id'] == opening_id]
            jds = [jd for jd in jds if jd['id'] == opening_id]
        
        # Lấy candidates
        all_candidates = []
        for opening in openings:
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
            "total_openings": len(openings),
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

