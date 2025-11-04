"""
FastAPI Backend - Trích xuất dữ liệu JD và CV từ Base Hiring API
"""
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
import os
from time import time
import pdfplumber
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google import genai
from google.genai import types
app = FastAPI(
    title="Base Hiring API - JD và CV Extractor",
    description="API để trích xuất dữ liệu JD (Job Description) và CV từ Base Hiring API",
    version="v2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Load from .env file
BASE_API_KEY = os.getenv('BASE_API_KEY')
if not BASE_API_KEY:
    raise ValueError("BASE_API_KEY chưa được cấu hình trong file .env")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY chưa được cấu hình trong file .env")

# Parse GEMINI_API_KEY_DU_PHONG từ string (comma-separated) sang list
GEMINI_API_KEY_DU_PHONG_STR = os.getenv('GEMINI_API_KEY_DU_PHONG', '')
GEMINI_API_KEY_DU_PHONG = [key.strip() for key in GEMINI_API_KEY_DU_PHONG_STR.split(',') if key.strip()] if GEMINI_API_KEY_DU_PHONG_STR else []

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
    """Trích xuất text từ PDF URL bằng pdfplumber (fallback method)"""
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

def extract_text_from_cv_url_with_genai(url):
    """Trích xuất text từ CV URL bằng Google Gemini AI"""
    if not url:
        return None
    
    # Thử với API key chính trước
    api_keys_to_try = [GEMINI_API_KEY] + GEMINI_API_KEY_DU_PHONG
    
    for idx, api_key in enumerate(api_keys_to_try):
        try:
            client = genai.Client(api_key=api_key)
            
            model = "gemini-flash-lite-latest"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"{url}\nĐọc text từ url"),
                    ],
                ),
            ]
            tools = [
                types.Tool(url_context=types.UrlContext()),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
                tools=tools,
                system_instruction=[
                    types.Part.from_text(text="Nội dung text trong link"),
                ],
            )
            
            # Thu thập tất cả text từ stream
            text_content = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    text_content += chunk.text
            
            if text_content.strip():
                return text_content.strip()
                
        except Exception as e:
            error_str = str(e).lower()
            error_repr = repr(e).lower()
            
            # Kiểm tra nếu là lỗi 429 (rate limit)
            is_rate_limit = (
                '429' in error_str or 
                '429' in error_repr or
                'rate limit' in error_str or
                'rate_limit' in error_str or
                'quota exceeded' in error_str or
                'resource exhausted' in error_str
            )
            
            # Nếu là lỗi 429 và còn API key khác, thử key tiếp theo
            if is_rate_limit and idx < len(api_keys_to_try) - 1:
                continue
            
            # Nếu không phải lỗi 429 hoặc đã hết key, tiếp tục thử key tiếp theo hoặc fallback
            continue
    
    # Nếu tất cả API keys đều fail, fallback về pdfplumber
    return extract_text_from_pdf(url)

def find_opening_id_by_name(query_name, api_key, similarity_threshold=0.5):
    """Tìm opening_id gần nhất với query_name bằng cosine similarity"""
    openings = get_base_openings(api_key, use_cache=True)
    
    if not openings:
        return None, None, 0.0
    
    # Nếu tìm thấy chính xác theo id hoặc name
    exact_match = next((op for op in openings if op['id'] == query_name or op['name'] == query_name), None)
    if exact_match:
        return exact_match['id'], exact_match['name'], 1.0
    
    # Nếu không tìm thấy chính xác, dùng cosine similarity
    opening_names = [op['name'] for op in openings]
    
    if not opening_names:
        return None, None, 0.0
    
    # Vectorize các tên opening
    vectorizer = TfidfVectorizer()
    try:
        name_vectors = vectorizer.fit_transform(opening_names)
        query_vector = vectorizer.transform([query_name])
        
        # Tính cosine similarity
        similarities = cosine_similarity(query_vector, name_vectors).flatten()
        
        # Tìm index có similarity cao nhất
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Nếu similarity >= threshold, trả về opening đó
        if best_similarity >= similarity_threshold:
            best_opening = openings[best_idx]
            return best_opening['id'], best_opening['name'], float(best_similarity)
        else:
            return None, None, float(best_similarity)
    except Exception:
        # Nếu có lỗi trong vectorization, trả về None
        return None, None, 0.0

def get_candidates_for_opening(opening_id, api_key, start_date=None, end_date=None, stage_name=None):
    """Truy xuất ứng viên cho một vị trí tuyển dụng cụ thể trong khoảng thời gian (luôn có cv_text)"""
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
        # Nếu có stage_name, tìm các stage name phù hợp bằng cosine similarity
        matching_stage_names = None
        if stage_name is not None:
            # Thu thập tất cả stage_name unique từ candidates
            all_stage_names = list(set([
                candidate.get('stage_name', '') 
                for candidate in data['candidates'] 
                if candidate.get('stage_name')
            ]))
            
            if not all_stage_names:
                # Nếu không có stage_name nào, lấy tất cả
                matching_stage_names = None
            else:
                # Kiểm tra exact match trước
                if stage_name in all_stage_names:
                    matching_stage_names = [stage_name]
                else:
                    # Dùng cosine similarity để tìm stage name gần nhất
                    try:
                        vectorizer = TfidfVectorizer()
                        stage_vectors = vectorizer.fit_transform(all_stage_names)
                        query_vector = vectorizer.transform([stage_name])
                        
                        similarities = cosine_similarity(query_vector, stage_vectors).flatten()
                        best_idx = np.argmax(similarities)
                        best_similarity = similarities[best_idx]
                        
                        # Nếu similarity >= 0.3, lấy stage name đó (ngưỡng thấp để bao quát hơn)
                        if best_similarity >= 0.3:
                            matching_stage_names = [all_stage_names[best_idx]]
                        else:
                            # Nếu không tìm thấy gì phù hợp, lấy tất cả
                            matching_stage_names = None
                    except Exception:
                        # Nếu có lỗi trong vectorization, lấy tất cả
                        matching_stage_names = None
        
        candidates = []
        for candidate in data['candidates']:
            # Lọc theo stage_name nếu có matching_stage_names
            if matching_stage_names is not None:
                candidate_stage_name = candidate.get('stage_name', '')
                if candidate_stage_name not in matching_stage_names:
                    continue
            
            cv_urls = candidate.get('cvs', [])
            cv_url = cv_urls[0] if isinstance(cv_urls, list) and len(cv_urls) > 0 else None
            
            # Luôn trích xuất cv_text từ CV URL bằng genai (fallback về pdfplumber nếu fail)
            cv_text = None
            if cv_url:
                cv_text = extract_text_from_cv_url_with_genai(cv_url)
            
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
                "cv_text": cv_text,  # Thay cv_url bằng cv_text
                "review": review,
                "form_data": form_data,
                "opening_id": opening_id,
                "stage_id": candidate.get('stage_id'),
                "stage_name": candidate.get('stage_name')
            }
            
            candidates.append(candidate_info)
        
        return candidates
    return []

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
    stage_id: Optional[str]
    stage_name: Optional[str]

# =================================================================
# API Endpoints
# =================================================================

@app.get("/", operation_id="healthCheck")
async def root():
    """Health check - Kiểm tra trạng thái API"""
    return {
        "status": "ok",
        "message": "Base Hiring API - Trích xuất JD và CV",
        "endpoints": {
            "get_candidates": "/api/opening/{opening_name_or_id}/candidates",
            "get_job_description": "/api/opening/{opening_name_or_id}/job-description"
        },
        "note": "Có thể sử dụng opening_name hoặc opening_id. Hệ thống sẽ tự động tìm opening gần nhất bằng cosine similarity nếu dùng name."
    }

@app.get("/api/opening/{opening_name_or_id}/job-description", operation_id="layJobDescriptionTheoOpening")
async def get_job_description_by_opening(
    opening_name_or_id: str = Path(..., description="Tên hoặc ID của vị trí tuyển dụng")
):
    """Lấy JD (Job Description) theo opening_name hoặc opening_id"""
    try:
        # Tìm opening_id từ name hoặc id bằng cosine similarity
        opening_id, matched_name, similarity_score = find_opening_id_by_name(
            opening_name_or_id, 
            BASE_API_KEY
        )
        
        if not opening_id:
            raise HTTPException(
                status_code=404, 
                detail=f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Similarity score cao nhất: {similarity_score:.2f}"
            )
        
        # Lấy JD (Job Description)
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            # Thử làm mới cache nếu không tìm thấy
            jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
            jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy JD cho vị trí '{opening_name_or_id}'"
            )
        
        return {
            "success": True,
            "query": opening_name_or_id,
            "opening_id": opening_id,
            "opening_name": matched_name,
            "similarity_score": similarity_score,
            "job_description": jd['job_description']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy JD: {str(e)}")

@app.get("/api/opening/{opening_name_or_id}/candidates", operation_id="layUngVienTheoOpening")
async def get_candidates_by_opening(
    opening_name_or_id: str = Path(..., description="Tên hoặc ID của vị trí tuyển dụng"),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    stage_name: Optional[str] = Query(None, description="Lọc ứng viên theo stage name. Bỏ trống để lấy tất cả.")
):
    """Lấy tất cả ứng viên theo opening_name hoặc opening_id (bao gồm cv_text)"""
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Ngày kết thúc phải sau ngày bắt đầu")
        
        # Tìm opening_id từ name hoặc id bằng cosine similarity
        opening_id, matched_name, similarity_score = find_opening_id_by_name(
            opening_name_or_id, 
            BASE_API_KEY
        )
        
        if not opening_id:
            raise HTTPException(
                status_code=404, 
                detail=f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Similarity score cao nhất: {similarity_score:.2f}"
            )
        
        candidates = get_candidates_for_opening(opening_id, BASE_API_KEY, start_date_obj, end_date_obj, stage_name)
        
        # Lấy JD (Job Description)
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            # Thử làm mới cache nếu không tìm thấy
            jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
            jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        job_description = jd['job_description'] if jd else None
        
        return {
            "success": True,
            "query": opening_name_or_id,
            "opening_id": opening_id,
            "opening_name": matched_name,
            "similarity_score": similarity_score,
            "job_description": job_description,
            "total_candidates": len(candidates),
            "candidates": candidates
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy ứng viên: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
