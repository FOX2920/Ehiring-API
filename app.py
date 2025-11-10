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
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google import genai
from google.genai import types
from pytz import timezone
import re
from html import unescape
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

# Google Sheet Script URL (optional) - để lấy dữ liệu bài test
GOOGLE_SHEET_SCRIPT_URL = os.getenv('GOOGLE_SHEET_SCRIPT_URL', None)

# Account API Key (optional) - để lấy thông tin users cho reviews
ACCOUNT_API_KEY = os.getenv('ACCOUNT_API_KEY', None)

# Cache configuration
CACHE_TTL = 300  # 5 phút cache
_cache = {
    'openings': {'data': None, 'timestamp': 0},
    'job_descriptions': {'data': None, 'timestamp': 0},
    'users_info': {'data': None, 'timestamp': 0}
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

def remove_html_tags(text):
    """Bỏ HTML tags và chuyển đổi thành text thuần túy"""
    if not text:
        return ""
    # Chuyển các thẻ <br> thành xuống dòng
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Bỏ tất cả các thẻ HTML còn lại
    text = re.sub(r'<[^>]+>', '', text)
    # Unescape các ký tự HTML entities (&lt;, &gt;, &amp;, etc.)
    text = unescape(text)
    # Loại bỏ các khoảng trắng thừa
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def get_users_info(use_cache=True):
    """Lấy thông tin users từ Account API và map username -> name + title (có cache)"""
    if not ACCOUNT_API_KEY:
        return {}
    
    current_time = time()
    
    # Kiểm tra cache
    if use_cache and _cache['users_info']['data'] is not None:
        if current_time - _cache['users_info']['timestamp'] < CACHE_TTL:
            return _cache['users_info']['data']
    
    try:
        users_url = "https://account.base.vn/extapi/v1/users"
        users_payload = {'access_token': ACCOUNT_API_KEY}
        users_response = requests.post(users_url, data=users_payload, timeout=10)
        users_response.raise_for_status()
        users_data = users_response.json()
        
        # Tạo dictionary để map username -> name + title
        username_to_info = {}
        if 'users' in users_data and isinstance(users_data['users'], list):
            for user in users_data['users']:
                username = user.get('username')
                name = user.get('name', '')
                title = user.get('title', '')
                if username:
                    # Nếu là Hoang Tran thì thay title thành CEO
                    if name == "Hoang Tran":
                        title = "CEO"
                    
                    # Kết hợp name và title
                    if title:
                        username_to_info[username] = {"name": name, "title": title}
                    else:
                        username_to_info[username] = {"name": name, "title": ""}
        
        # Lưu vào cache
        if use_cache:
            _cache['users_info'] = {'data': username_to_info, 'timestamp': current_time}
        
        return username_to_info
    except Exception as e:
        # Nếu có lỗi, trả về dict rỗng
        return {}

def process_evaluations(evaluations):
    """Xử lý evaluations và trả về danh sách reviews với đầy đủ thông tin (tên, chức danh, nội dung)"""
    if not isinstance(evaluations, list) or len(evaluations) == 0:
        return []
    
    # Lấy thông tin users
    username_to_info = get_users_info(use_cache=True)
    
    reviews = []
    for eval_item in evaluations:
        if 'content' in eval_item:
            # Bỏ HTML tags
            clean_content = remove_html_tags(eval_item.get('content', ''))
            
            # Lấy username và chuyển thành tên thật + chức danh
            username = eval_item.get('username')
            user_info = username_to_info.get(username, {}) if username else {}
            name = user_info.get('name', username) if user_info else (username if username else "N/A")
            title = user_info.get('title', '') if user_info else ''
            
            review = {
                "id": eval_item.get('id'),
                "name": name,
                "title": title,
                "content": clean_content
            }
            reviews.append(review)
    
    return reviews

def extract_text_from_pdf(url=None, file_bytes=None):
    """Trích xuất text từ PDF URL hoặc file bytes bằng pdfplumber"""
    pdf_file = None
    if file_bytes:
        pdf_file = file_bytes
    elif url:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pdf_file = BytesIO(response.content)
        except Exception:
            return None
    else:
        return None
    
    try:
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

def extract_text_from_docx(file_bytes):
    """Trích xuất text từ DOCX file bytes"""
    if not DOCX_AVAILABLE:
        return None
    try:
        doc = Document(file_bytes)
        text = "\n".join([p.text for p in doc.paragraphs]).strip()
        return text if text else None
    except Exception as e:
        return None

def download_file_to_bytes(url):
    """Tải file từ URL và trả về BytesIO"""
    if not url:
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception:
        return None

def is_target_file(url, name):
    """Kiểm tra xem file có phải PDF/DOCX/DOC không"""
    if not url or not name:
        return False
    url_low = url.lower().split('?')[0]
    name_low = name.lower()
    return url_low.endswith(('.pdf', '.docx', '.doc')) or name_low.endswith(('.pdf', '.docx', '.doc'))

def find_files_in_html(html_content):
    """Tìm các file PDF/DOCX/DOC trong HTML content"""
    found = []
    if not html_content:
        return found
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            name = a.get_text().strip() or href.split('/')[-1]
            if is_target_file(href, name):
                found.append((href, name))
        return found
    except Exception:
        return found

def get_offer_letter(candidate_id, api_key):
    """Lấy offer letter từ messages API của ứng viên"""
    if not candidate_id or not api_key:
        return None
    
    try:
        url = "https://hiring.base.vn/publicapi/v2/candidate/messages"
        payload = {
            'access_token': api_key,
            'id': candidate_id
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'messages' not in data:
            return None
        
        messages = data['messages']
        if not messages:
            return None
        
        # Duyệt từ tin nhắn mới nhất về cũ nhất
        for msg in messages:
            # Ưu tiên tìm trong attachments
            priority_files = []
            if msg.get('has_attachment', 0) > 0:
                for att in msg.get('attachments', []):
                    url_att = att.get('src') or att.get('url') or att.get('org')
                    name_att = att.get('name', 'unknown')
                    if url_att and is_target_file(url_att, name_att):
                        priority_files.append((url_att, name_att))
            
            # Nếu không có trong attachments, tìm trong HTML content
            secondary_files = []
            if not priority_files:
                secondary_files = find_files_in_html(msg.get('content', ''))
            
            all_files = priority_files + secondary_files
            if not all_files:
                continue
            
            # Thử tải và trích xuất file đầu tiên tìm được
            for file_url, file_name in all_files:
                file_bytes = download_file_to_bytes(file_url)
                if not file_bytes:
                    continue
                
                ext = file_name.lower().split('.')[-1] if '.' in file_name else file_url.split('.')[-1].split('?')[0].lower()
                text = None
                
                if 'pdf' in ext:
                    text = extract_text_from_pdf(file_bytes=file_bytes)
                elif 'docx' in ext and DOCX_AVAILABLE:
                    text = extract_text_from_docx(file_bytes)
                elif 'doc' == ext:
                    text = None  # File .doc cũ, không hỗ trợ
                
                if text:
                    return {
                        "url": file_url,
                        "name": file_name,
                        "text": text
                    }
        
        return None
    except Exception as e:
        # Nếu có lỗi, trả về None (không làm gián đoạn flow chính)
        return None

def extract_text_from_cv_url_with_genai(url):
    """Trích xuất text từ CV URL, ưu tiên pdfplumber, fallback về Google Gemini AI"""
    if not url:
        return None
    
    # Tạm thời ưu tiên sử dụng pdfplumber trước
    pdf_text = extract_text_from_pdf(url)
    if pdf_text:
        return pdf_text
    
    # Nếu pdfplumber không thành công, fallback về Gemini AI
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
            
            # Nếu không phải lỗi 429 hoặc đã hết key, tiếp tục thử key tiếp theo
            continue
    
    # Nếu tất cả đều fail, trả về None
    return None

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
        
        # Bước 1: Lọc ứng viên theo stage_name trước (chưa trích xuất cv_text để tiết kiệm request)
        filtered_candidates = []
        for candidate in data['candidates']:
            # Lọc theo stage_name nếu có matching_stage_names
            if matching_stage_names is not None:
                candidate_stage_name = candidate.get('stage_name', '')
                if candidate_stage_name not in matching_stage_names:
                    continue
            
            filtered_candidates.append(candidate)
        
        # Bước 2: Trích xuất cv_text chỉ cho các ứng viên đã được lọc
        candidates = []
        for candidate in filtered_candidates:
            cv_urls = candidate.get('cvs', [])
            cv_url = cv_urls[0] if isinstance(cv_urls, list) and len(cv_urls) > 0 else None
            
            # Chỉ trích xuất cv_text từ CV URL sau khi đã lọc xong (tiết kiệm request Gemini)
            cv_text = None
            if cv_url:
                cv_text = extract_text_from_cv_url_with_genai(cv_url)
            
            # Xử lý evaluations để lấy reviews chi tiết
            reviews = process_evaluations(candidate.get('evaluations', []))
            # Giữ lại review cũ (text đơn giản) để tương thích ngược
            review = extract_message(candidate.get('evaluations', []))
            
            form_data = {}
            if 'form' in candidate and isinstance(candidate['form'], list):
                for item in candidate['form']:
                    if isinstance(item, dict) and 'id' in item and 'value' in item:
                        form_data[item['id']] = item['value']
            
            # Lấy dữ liệu bài test từ Google Sheet
            test_results = get_test_results_from_google_sheet(candidate.get('id'))
            
            # Lấy offer letter nếu có
            offer_letter = get_offer_letter(candidate.get('id'), api_key)
            
            candidate_info = {
                "id": candidate.get('id'),
                "name": candidate.get('name'),
                "email": candidate.get('email'),
                "phone": candidate.get('phone'),
                "gender": candidate.get('gender'),
                "cv_url": cv_url,
                "cv_text": cv_text,
                "review": review,  # Giữ lại để tương thích ngược
                "reviews": reviews,  # Danh sách reviews chi tiết với tên, chức danh, nội dung
                "form_data": form_data,
                "opening_id": opening_id,
                "stage_id": candidate.get('stage_id'),
                "stage_name": candidate.get('stage_name'),
                "test_results": test_results,
                "offer_letter": offer_letter
            }
            
            candidates.append(candidate_info)
        
        return candidates
    return []

def get_interviews(api_key, start_date=None, end_date=None, opening_id=None, filter_date=None):
    """Truy xuất lịch phỏng vấn từ Base API, chỉ trả về các trường quan trọng. Lọc dựa trên date của time_dt nếu có filter_date."""
    url = "https://hiring.base.vn/publicapi/v2/interview/list"
    
    payload = {
        'access_token': api_key,
    }
    
    # Không truyền start_date/end_date vào API Base, sẽ lọc sau khi chuyển đổi time_dt
    # Chỉ dùng để API Base lọc sơ bộ nếu cần
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Lỗi kết nối đến Base API khi lấy lịch phỏng vấn: {e}")

    data = response.json()
    if 'interviews' in data and data['interviews']:
        interviews = data['interviews']
        
        # Nếu có opening_id, lọc theo opening_id
        if opening_id:
            interviews = [
                interview for interview in interviews
                if interview.get('opening_id') == opening_id
            ]
        
        # Xử lý và chỉ lấy các trường quan trọng
        processed_interviews = []
        hcm_tz = timezone('Asia/Ho_Chi_Minh')
        
        for interview in interviews:
            # Chỉ lấy các trường quan trọng
            processed_interview = {
                'id': interview.get('id'),
                'candidate_id': interview.get('candidate_id'),
                'candidate_name': interview.get('candidate_name'),
                'opening_name': interview.get('opening_name'),
                'time_dt': None
            }
            
            # Chuyển đổi timestamp 'time' sang datetime với timezone Asia/Ho_Chi_Minh
            time_dt_date = None
            if 'time' in interview and interview.get('time'):
                try:
                    timestamp = int(interview['time'])
                    dt = datetime.fromtimestamp(timestamp, tz=timezone('UTC'))
                    dt_hcm = dt.astimezone(hcm_tz)
                    processed_interview['time_dt'] = dt_hcm.isoformat()
                    time_dt_date = dt_hcm.date()  # Lấy date để lọc
                except (ValueError, TypeError, OSError):
                    pass
            
            # Lọc dựa trên date của time_dt
            if filter_date:
                if time_dt_date is None or time_dt_date != filter_date:
                    continue  # Bỏ qua nếu không có time_dt hoặc date không khớp
            
            processed_interviews.append(processed_interview)
        
        return processed_interviews
    
    return []

def get_test_results_from_google_sheet(candidate_id):
    """Lấy dữ liệu bài test của ứng viên từ Google Sheet"""
    if not GOOGLE_SHEET_SCRIPT_URL:
        return None
    
    try:
        payload = {
            'action': 'read_data',
            'filters': {
                'candidate_id': str(candidate_id)
            }
        }
        
        response = requests.post(
            GOOGLE_SHEET_SCRIPT_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success') and result.get('data'):
            # Trả về danh sách bài test của ứng viên
            test_results = []
            for item in result.get('data', []):
                test_result = {
                    'test_name': item.get('Tên bài test', ''),
                    'score': item.get('Score', ''),
                    'time': item.get('Time', ''),
                    'link': item.get('Link', ''),
                    'test_content': item.get('test content', '')
                }
                test_results.append(test_result)
            return test_results if test_results else None
        
        return None
    except Exception as e:
        # Nếu có lỗi, trả về None (không làm gián đoạn flow chính)
        return None

def get_candidate_details(candidate_id, api_key):
    """Lấy và xử lý dữ liệu chi tiết ứng viên từ API Base.vn, trả về JSON phẳng"""
    url = "https://hiring.base.vn/publicapi/v2/candidate/get"
    
    payload = {
        'access_token': api_key,
        'id': candidate_id
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Lỗi kết nối đến Base API khi lấy chi tiết ứng viên: {e}")
    
    raw_response = response.json()
    
    # Kiểm tra API có trả về lỗi logic không (vd: 'code': 1 là thành công)
    if raw_response.get('code') != 1 or not raw_response.get('candidate'):
        raise HTTPException(
            status_code=404, 
            detail=f"Không tìm thấy ứng viên với ID '{candidate_id}'. {raw_response.get('message', '')}"
        )
    
    # Lấy dữ liệu gốc của ứng viên
    candidate_data = raw_response.get('candidate', {})
    
    # Hàm trợ giúp để "làm phẳng" các danh sách lồng nhau
    def flatten_fields(field_list):
        """Chuyển đổi danh sách [{'id': 'key1', 'value': 'val1'}, ...] thành {'key1': 'val1', ...}"""
        flat_dict = {}
        if isinstance(field_list, list):
            for item in field_list:
                if isinstance(item, dict) and 'id' in item:
                    flat_dict[item['id']] = item.get('value')
        return flat_dict
    
    # Bắt đầu với các trường dữ liệu chính
    refined_data = {
        'id': candidate_data.get('id'),
        'ten': candidate_data.get('name'),
        'email': candidate_data.get('email'),
        'so_dien_thoai': candidate_data.get('phone'),
        
        # Lấy tên vị trí tuyển dụng chính xác từ 'evaluations'
        'vi_tri_ung_tuyen': (candidate_data.get('evaluations') or [{}])[0].get('opening_export', {}).get('name', candidate_data.get('title')),
        
        # Lấy opening_id từ evaluations nếu có
        'opening_id': (candidate_data.get('evaluations') or [{}])[0].get('opening_export', {}).get('id'),
        
        # Lấy stage_id và stage_name
        'stage_id': candidate_data.get('stage_id'),
        'stage_name': candidate_data.get('stage_name', candidate_data.get('status')),
        
        'nguon_ung_vien': candidate_data.get('source'),
        'ngay_sinh': candidate_data.get('dob'),
        'gioi_tinh': candidate_data.get('gender_text'),
        'dia_chi_hien_tai': candidate_data.get('address'),
        'cccd': candidate_data.get('ssn'),
        'cv_url': (candidate_data.get('cvs') or [None])[0]
    }
    
    # Xử lý và gộp dữ liệu từ 'fields' và 'form'
    field_data = flatten_fields(candidate_data.get('fields', []))
    form_data = flatten_fields(candidate_data.get('form', []))
    
    # Cập nhật vào dict chính
    refined_data.update(field_data)
    refined_data.update(form_data)
    
    # Xử lý evaluations để lấy reviews chi tiết
    reviews = process_evaluations(candidate_data.get('evaluations', []))
    refined_data['reviews'] = reviews
    
    # Lấy offer letter nếu có
    offer_letter = get_offer_letter(candidate_id, api_key)
    refined_data['offer_letter'] = offer_letter
    
    return refined_data

# =================================================================
# Request/Response Models
# =================================================================

class JobDescriptionResponse(BaseModel):
    id: str
    name: str
    job_description: str

class TestResult(BaseModel):
    test_name: Optional[str]
    score: Optional[str]
    time: Optional[str]
    link: Optional[str]
    test_content: Optional[str]

class Review(BaseModel):
    id: Optional[str]
    name: str
    title: str
    content: str

class OfferLetter(BaseModel):
    url: Optional[str]
    name: Optional[str]
    text: Optional[str]

class CandidateResponse(BaseModel):
    id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    gender: Optional[str]
    cv_url: Optional[str]
    cv_text: Optional[str]
    review: Optional[str]
    reviews: Optional[list[Review]]
    form_data: dict
    opening_id: str
    stage_id: Optional[str]
    stage_name: Optional[str]
    test_results: Optional[list[TestResult]]
    offer_letter: Optional[OfferLetter]

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
            "get_job_description": "/api/opening/{opening_name_or_id}/job-description",
            "get_interviews": "/api/interviews",
            "get_candidate_details": "/api/candidate/{candidate_id}"
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

@app.get("/api/interviews", operation_id="layLichPhongVan")
async def get_interviews_by_opening(
    opening_name_or_id: Optional[str] = Query(None, description="Tên hoặc ID của vị trí tuyển dụng để lọc. Bỏ trống để lấy tất cả."),
    date: Optional[str] = Query(None, description="Lấy lịch phỏng vấn cho 1 ngày cụ thể (YYYY-MM-DD). Nếu có tham số này, sẽ bỏ qua start_date và end_date."),
    start_date: Optional[str] = Query(None, description="Ngày bắt đầu lọc lịch phỏng vấn (YYYY-MM-DD). Bỏ trống để lấy tất cả."),
    end_date: Optional[str] = Query(None, description="Ngày kết thúc lọc lịch phỏng vấn (YYYY-MM-DD). Bỏ trống để lấy tất cả.")
):
    """Lấy lịch phỏng vấn, có thể lọc theo opening_name hoặc opening_id (tự động tìm bằng cosine similarity). Có thể lấy cho 1 ngày cụ thể bằng tham số date. Lọc dựa trên date của time_dt."""
    try:
        filter_date_obj = None
        
        # Nếu có tham số date, dùng nó để lọc dựa trên time_dt
        if date:
            filter_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        
        opening_id = None
        matched_name = None
        similarity_score = None
        
        # Nếu có opening_name_or_id, tìm opening_id bằng cosine similarity
        if opening_name_or_id:
            opening_id, matched_name, similarity_score = find_opening_id_by_name(
                opening_name_or_id,
                BASE_API_KEY
            )
            
            if not opening_id:
                # Nếu không tìm thấy, vẫn trả về tất cả interviews nhưng có thông báo
                opening_id = None
        
        # Lấy tất cả interviews và lọc dựa trên date của time_dt
        interviews = get_interviews(BASE_API_KEY, opening_id=opening_id, filter_date=filter_date_obj)
        
        return {
            "success": True,
            "query": opening_name_or_id,
            "date": date,
            "opening_id": opening_id,
            "opening_name": matched_name,
            "similarity_score": similarity_score,
            "total_interviews": len(interviews),
            "interviews": interviews
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Định dạng ngày không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy lịch phỏng vấn: {str(e)}")

@app.get("/api/candidate/{candidate_id}", operation_id="layChiTietUngVien")
async def get_candidate_details_endpoint(
    candidate_id: str = Path(..., description="ID của ứng viên")
):
    """Lấy chi tiết ứng viên theo candidate_id. Tự động trích xuất cv_text từ cv_url bằng Gemini AI và thêm JD dựa trên opening name."""
    try:
        # Lấy dữ liệu chi tiết ứng viên
        candidate_data = get_candidate_details(candidate_id, BASE_API_KEY)
        
        # Trích xuất cv_text từ cv_url nếu có
        cv_url = candidate_data.get('cv_url')
        cv_text = None
        if cv_url:
            cv_text = extract_text_from_cv_url_with_genai(cv_url)
            candidate_data['cv_text'] = cv_text
        
        # Lấy dữ liệu bài test từ Google Sheet
        test_results = get_test_results_from_google_sheet(candidate_id)
        candidate_data['test_results'] = test_results
        
        # Lấy JD dựa trên opening name
        opening_name = candidate_data.get('vi_tri_ung_tuyen')
        opening_id = candidate_data.get('opening_id')
        job_description = None
        
        if opening_name or opening_id:
            # Tìm opening_id nếu chỉ có opening_name
            if not opening_id and opening_name:
                opening_id, matched_name, similarity_score = find_opening_id_by_name(
                    opening_name,
                    BASE_API_KEY
                )
            
            # Lấy JD nếu có opening_id
            if opening_id:
                jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
                jd = next((jd for jd in jds if jd['id'] == opening_id), None)
                
                if not jd:
                    # Thử làm mới cache nếu không tìm thấy
                    jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
                    jd = next((jd for jd in jds if jd['id'] == opening_id), None)
                
                if jd:
                    job_description = jd['job_description']
                    candidate_data['job_description'] = job_description
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "candidate_details": candidate_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy chi tiết ứng viên: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
