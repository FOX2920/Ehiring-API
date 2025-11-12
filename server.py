"""
FastMCP Server - Trích xuất dữ liệu JD và CV từ Base Hiring API
"""
from mcp.server.fastmcp import FastMCP
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

# Khởi tạo FastMCP server
mcp = FastMCP("Base Hiring MCP Server")

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

# Google Sheet Script URL (optional)
GOOGLE_SHEET_SCRIPT_URL = os.getenv('GOOGLE_SHEET_SCRIPT_URL', None)

# Account API Key (optional)
ACCOUNT_API_KEY = os.getenv('ACCOUNT_API_KEY', None)

# Cache configuration
CACHE_TTL = 300  # 5 phút cache
_cache = {
    'openings': {'data': None, 'timestamp': 0},
    'job_descriptions': {'data': None, 'timestamp': 0},
    'users_info': {'data': None, 'timestamp': 0}
}

# =================================================================
# Helper Functions
# =================================================================

def get_base_openings(api_key, use_cache=True):
    """Truy xuất vị trí tuyển dụng đang hoạt động từ Base API (có cache)"""
    current_time = time()
    if not api_key:
        raise Exception("BASE_API_KEY chưa được cấu hình")
    
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
        raise Exception(f"Lỗi kết nối đến Base API: {e}")

    data = response.json()
    openings = data.get('openings', [])
    
    filtered_openings = [
        {"id": opening['id'], "name": opening['name']}
        for opening in openings
        if opening.get('status') == '10'
    ]
    
    if use_cache:
        _cache['openings'] = {'data': filtered_openings, 'timestamp': current_time}
    
    return filtered_openings

def get_job_descriptions(api_key, use_cache=True):
    """Truy xuất JD (Job Description) từ các vị trí tuyển dụng đang mở (có cache)"""
    current_time = time()
    if not api_key:
        raise Exception("BASE_API_KEY chưa được cấu hình")

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
        raise Exception(f"Lỗi kết nối đến Base API: {e}")

    data = response.json()
    
    if 'openings' in data:
        openings = data['openings']
        results = []
        
        for opening in openings:
            if opening.get('status') == '10':
                html_content = opening.get('content', '')
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text()
                
                if len(text_content) >= 10:
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
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def get_users_info(use_cache=True):
    """Lấy thông tin users từ Account API và map username -> name + title (có cache)"""
    if not ACCOUNT_API_KEY:
        return {}
    
    current_time = time()
    
    if use_cache and _cache['users_info']['data'] is not None:
        if current_time - _cache['users_info']['timestamp'] < CACHE_TTL:
            return _cache['users_info']['data']
    
    try:
        users_url = "https://account.base.vn/extapi/v1/users"
        users_payload = {'access_token': ACCOUNT_API_KEY}
        users_response = requests.post(users_url, data=users_payload, timeout=10)
        users_response.raise_for_status()
        users_data = users_response.json()
        
        username_to_info = {}
        if 'users' in users_data and isinstance(users_data['users'], list):
            for user in users_data['users']:
                username = user.get('username')
                name = user.get('name', '')
                title = user.get('title', '')
                if username:
                    if name == "Hoang Tran":
                        title = "CEO"
                    
                    if title:
                        username_to_info[username] = {"name": name, "title": title}
                    else:
                        username_to_info[username] = {"name": name, "title": ""}
        
        if use_cache:
            _cache['users_info'] = {'data': username_to_info, 'timestamp': current_time}
        
        return username_to_info
    except Exception:
        return {}

def process_evaluations(evaluations):
    """Xử lý evaluations và trả về danh sách reviews với đầy đủ thông tin"""
    if not isinstance(evaluations, list) or len(evaluations) == 0:
        return []
    
    username_to_info = get_users_info(use_cache=True)
    
    reviews = []
    for eval_item in evaluations:
        if 'content' in eval_item:
            clean_content = remove_html_tags(eval_item.get('content', ''))
            
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
    except Exception:
        return None

def extract_text_from_docx(file_bytes):
    """Trích xuất text từ DOCX file bytes"""
    if not DOCX_AVAILABLE:
        return None
    try:
        doc = Document(file_bytes)
        text = "\n".join([p.text for p in doc.paragraphs]).strip()
        return text if text else None
    except Exception:
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
        
        for msg in messages:
            priority_files = []
            if msg.get('has_attachment', 0) > 0:
                for att in msg.get('attachments', []):
                    url_att = att.get('src') or att.get('url') or att.get('org')
                    name_att = att.get('name', 'unknown')
                    if url_att and is_target_file(url_att, name_att):
                        priority_files.append((url_att, name_att))
            
            secondary_files = []
            if not priority_files:
                secondary_files = find_files_in_html(msg.get('content', ''))
            
            all_files = priority_files + secondary_files
            if not all_files:
                continue
            
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
                    text = None
                
                if text:
                    return {
                        "url": file_url,
                        "name": file_name,
                        "text": text
                    }
        
        return None
    except Exception:
        return None

def extract_text_from_cv_url_with_genai(url):
    """Trích xuất text từ CV URL, ưu tiên pdfplumber, fallback về Google Gemini AI"""
    if not url:
        return None
    
    pdf_text = extract_text_from_pdf(url)
    if pdf_text:
        return pdf_text
    
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
            
            is_rate_limit = (
                '429' in error_str or 
                '429' in error_repr or
                'rate limit' in error_str or
                'rate_limit' in error_str or
                'quota exceeded' in error_str or
                'resource exhausted' in error_str
            )
            
            if is_rate_limit and idx < len(api_keys_to_try) - 1:
                continue
            
            continue
    
    return None

def find_opening_id_by_name(query_name, api_key, similarity_threshold=0.5):
    """Tìm opening_id gần nhất với query_name bằng cosine similarity"""
    openings = get_base_openings(api_key, use_cache=True)
    
    if not openings:
        return None, None, 0.0
    
    exact_match = next((op for op in openings if op['id'] == query_name or op['name'] == query_name), None)
    if exact_match:
        return exact_match['id'], exact_match['name'], 1.0
    
    opening_names = [op['name'] for op in openings]
    
    if not opening_names:
        return None, None, 0.0
    
    vectorizer = TfidfVectorizer()
    try:
        name_vectors = vectorizer.fit_transform(opening_names)
        query_vector = vectorizer.transform([query_name])
        
        similarities = cosine_similarity(query_vector, name_vectors).flatten()
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= similarity_threshold:
            best_opening = openings[best_idx]
            return best_opening['id'], best_opening['name'], float(best_similarity)
        else:
            return None, None, float(best_similarity)
    except Exception:
        return None, None, 0.0

def find_candidate_by_name_in_opening(candidate_name, opening_id, api_key, similarity_threshold=0.5, filter_stages=None):
    """Tìm candidate_id dựa trên tên ứng viên trong một opening cụ thể bằng cosine similarity"""
    if not candidate_name or not opening_id:
        return None, 0.0
    
    url = "https://hiring.base.vn/publicapi/v2/candidate/list"
    payload = {
        'access_token': api_key,
        'opening_id': opening_id,
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None, 0.0
    
    data = response.json()
    if 'candidates' not in data or not data['candidates']:
        return None, 0.0
    
    if filter_stages:
        filtered_candidates = []
        for candidate in data['candidates']:
            stage_name = candidate.get('stage_name', '')
            if stage_name and stage_name in filter_stages:
                filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            return None, 0.0
    else:
        filtered_candidates = data['candidates']
    
    candidate_names = [c.get('name', '') for c in filtered_candidates if c.get('name')]
    
    if not candidate_names:
        return None, 0.0
    
    exact_match = next((c for c in filtered_candidates if c.get('name') == candidate_name), None)
    if exact_match:
        return exact_match.get('id'), 1.0
    
    try:
        vectorizer = TfidfVectorizer()
        name_vectors = vectorizer.fit_transform(candidate_names)
        query_vector = vectorizer.transform([candidate_name])
        
        similarities = cosine_similarity(query_vector, name_vectors).flatten()
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= similarity_threshold:
            candidates_with_names = [c for c in filtered_candidates if c.get('name')]
            if best_idx < len(candidates_with_names):
                best_candidate = candidates_with_names[best_idx]
                return best_candidate.get('id'), float(best_similarity)
        
        return None, float(best_similarity)
    except Exception:
        return None, 0.0

def get_candidates_for_opening(opening_id, api_key, start_date=None, end_date=None, stage_name=None):
    """Truy xuất ứng viên cho một vị trí tuyển dụng cụ thể"""
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
        raise Exception(f"Lỗi kết nối đến Base API khi lấy ứng viên: {e}")

    data = response.json()
    if 'candidates' in data and data['candidates']:
        matching_stage_names = None
        if stage_name is not None:
            all_stage_names = list(set([
                candidate.get('stage_name', '') 
                for candidate in data['candidates'] 
                if candidate.get('stage_name')
            ]))
            
            if not all_stage_names:
                matching_stage_names = None
            else:
                if stage_name in all_stage_names:
                    matching_stage_names = [stage_name]
                else:
                    try:
                        vectorizer = TfidfVectorizer()
                        stage_vectors = vectorizer.fit_transform(all_stage_names)
                        query_vector = vectorizer.transform([stage_name])
                        
                        similarities = cosine_similarity(query_vector, stage_vectors).flatten()
                        best_idx = np.argmax(similarities)
                        best_similarity = similarities[best_idx]
                        
                        if best_similarity >= 0.3:
                            matching_stage_names = [all_stage_names[best_idx]]
                        else:
                            matching_stage_names = None
                    except Exception:
                        matching_stage_names = None
        
        filtered_candidates = []
        for candidate in data['candidates']:
            if matching_stage_names is not None:
                candidate_stage_name = candidate.get('stage_name', '')
                if candidate_stage_name not in matching_stage_names:
                    continue
            
            filtered_candidates.append(candidate)
        
        candidates = []
        for candidate in filtered_candidates:
            cv_urls = candidate.get('cvs', [])
            cv_url = cv_urls[0] if isinstance(cv_urls, list) and len(cv_urls) > 0 else None
            
            cv_text = None
            if cv_url:
                cv_text = extract_text_from_cv_url_with_genai(cv_url)
            
            reviews = process_evaluations(candidate.get('evaluations', []))
            review = extract_message(candidate.get('evaluations', []))
            
            form_data = {}
            if 'form' in candidate and isinstance(candidate['form'], list):
                for item in candidate['form']:
                    if isinstance(item, dict) and 'id' in item and 'value' in item:
                        form_data[item['id']] = item['value']
            
            test_results = get_test_results_from_google_sheet(candidate.get('id'))
            
            candidate_info = {
                "id": candidate.get('id'),
                "name": candidate.get('name'),
                "email": candidate.get('email'),
                "phone": candidate.get('phone'),
                "gender": candidate.get('gender'),
                "cv_url": cv_url,
                "cv_text": cv_text,
                "review": review,
                "reviews": reviews,
                "form_data": form_data,
                "opening_id": opening_id,
                "stage_id": candidate.get('stage_id'),
                "stage_name": candidate.get('stage_name'),
                "test_results": test_results
            }
            
            candidates.append(candidate_info)
        
        return candidates
    return []

def get_interviews(api_key, start_date=None, end_date=None, opening_id=None, filter_date=None):
    """Truy xuất lịch phỏng vấn từ Base API"""
    url = "https://hiring.base.vn/publicapi/v2/interview/list"
    
    payload = {
        'access_token': api_key,
    }
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Lỗi kết nối đến Base API khi lấy lịch phỏng vấn: {e}")

    data = response.json()
    if 'interviews' in data and data['interviews']:
        interviews = data['interviews']
        
        if opening_id:
            interviews = [
                interview for interview in interviews
                if interview.get('opening_id') == opening_id
            ]
        
        processed_interviews = []
        hcm_tz = timezone('Asia/Ho_Chi_Minh')
        
        for interview in interviews:
            processed_interview = {
                'id': interview.get('id'),
                'candidate_id': interview.get('candidate_id'),
                'candidate_name': interview.get('candidate_name'),
                'opening_name': interview.get('opening_name'),
                'time_dt': None
            }
            
            time_dt_date = None
            if 'time' in interview and interview.get('time'):
                try:
                    timestamp = int(interview['time'])
                    dt = datetime.fromtimestamp(timestamp, tz=timezone('UTC'))
                    dt_hcm = dt.astimezone(hcm_tz)
                    processed_interview['time_dt'] = dt_hcm.isoformat()
                    time_dt_date = dt_hcm.date()
                except (ValueError, TypeError, OSError):
                    pass
            
            if filter_date:
                if time_dt_date is None or time_dt_date != filter_date:
                    continue
            
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
    except Exception:
        return None

def get_candidate_details(candidate_id, api_key):
    """Lấy và xử lý dữ liệu chi tiết ứng viên từ API Base.vn"""
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
        raise Exception(f"Lỗi kết nối đến Base API khi lấy chi tiết ứng viên: {e}")
    
    raw_response = response.json()
    
    if raw_response.get('code') != 1 or not raw_response.get('candidate'):
        raise Exception(f"Không tìm thấy ứng viên với ID '{candidate_id}'. {raw_response.get('message', '')}")
    
    candidate_data = raw_response.get('candidate', {})
    
    def flatten_fields(field_list):
        flat_dict = {}
        if isinstance(field_list, list):
            for item in field_list:
                if isinstance(item, dict) and 'id' in item:
                    flat_dict[item['id']] = item.get('value')
        return flat_dict
    
    refined_data = {
        'id': candidate_data.get('id'),
        'ten': candidate_data.get('name'),
        'email': candidate_data.get('email'),
        'so_dien_thoai': candidate_data.get('phone'),
        'vi_tri_ung_tuyen': (candidate_data.get('evaluations') or [{}])[0].get('opening_export', {}).get('name', candidate_data.get('title')),
        'opening_id': (candidate_data.get('evaluations') or [{}])[0].get('opening_export', {}).get('id'),
        'stage_id': candidate_data.get('stage_id'),
        'stage_name': candidate_data.get('stage_name', candidate_data.get('status')),
        'nguon_ung_vien': candidate_data.get('source'),
        'ngay_sinh': candidate_data.get('dob'),
        'gioi_tinh': candidate_data.get('gender_text'),
        'dia_chi_hien_tai': candidate_data.get('address'),
        'cccd': candidate_data.get('ssn'),
        'cv_url': (candidate_data.get('cvs') or [None])[0]
    }
    
    field_data = flatten_fields(candidate_data.get('fields', []))
    form_data = flatten_fields(candidate_data.get('form', []))
    
    refined_data.update(field_data)
    refined_data.update(form_data)
    
    reviews = process_evaluations(candidate_data.get('evaluations', []))
    refined_data['reviews'] = reviews
    
    return refined_data

# =================================================================
# MCP Tools (FastMCP endpoints)
# =================================================================

@mcp.tool()
def lay_job_description_theo_opening(
    opening_name_or_id: Optional[str] = None
) -> dict:
    """Lấy JD (Job Description) theo opening_name hoặc opening_id. 
    Nếu không có tham số hoặc không tìm thấy, trả về tất cả các opening có status 10.
    
    Args:
        opening_name_or_id: Tên hoặc ID của vị trí tuyển dụng. Bỏ trống để lấy tất cả.
    """
    try:
        openings = get_base_openings(BASE_API_KEY, use_cache=True)
        
        if not opening_name_or_id:
            return {
                "success": True,
                "query": None,
                "message": "Trả về tất cả các opening có status 10.",
                "total_openings": len(openings),
                "openings": openings
            }
        
        opening_id, matched_name, similarity_score = find_opening_id_by_name(
            opening_name_or_id, 
            BASE_API_KEY
        )
        
        if not opening_id:
            return {
                "success": True,
                "query": opening_name_or_id,
                "message": f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Trả về tất cả các opening có status 10.",
                "similarity_score": similarity_score,
                "total_openings": len(openings),
                "openings": openings
            }
        
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
            jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
            return {
                "success": True,
                "query": opening_name_or_id,
                "opening_id": opening_id,
                "opening_name": matched_name,
                "similarity_score": similarity_score,
                "message": f"Không tìm thấy JD cho vị trí '{opening_name_or_id}'. Trả về tất cả các opening có status 10.",
                "total_openings": len(openings),
                "openings": openings
            }
        
        return {
            "success": True,
            "query": opening_name_or_id,
            "opening_id": opening_id,
            "opening_name": matched_name,
            "similarity_score": similarity_score,
            "job_description": jd['job_description']
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def lay_ung_vien_theo_opening(
    opening_name_or_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stage_name: Optional[str] = None
) -> dict:
    """Lấy tất cả ứng viên theo opening_name hoặc opening_id (bao gồm cv_text).
    
    Args:
        opening_name_or_id: Tên hoặc ID của vị trí tuyển dụng
        start_date: Ngày bắt đầu lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả.
        end_date: Ngày kết thúc lọc ứng viên (YYYY-MM-DD). Bỏ trống để lấy tất cả.
        stage_name: Lọc ứng viên theo stage name. Bỏ trống để lấy tất cả.
    """
    try:
        start_date_obj, end_date_obj = None, None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            return {"success": False, "error": "Ngày kết thúc phải sau ngày bắt đầu"}
        
        opening_id, matched_name, similarity_score = find_opening_id_by_name(
            opening_name_or_id, 
            BASE_API_KEY
        )
        
        if not opening_id:
            return {
                "success": False,
                "error": f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Similarity score cao nhất: {similarity_score:.2f}"
            }
        
        candidates = get_candidates_for_opening(opening_id, BASE_API_KEY, start_date_obj, end_date_obj, stage_name)
        
        jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
        jd = next((jd for jd in jds if jd['id'] == opening_id), None)
        
        if not jd:
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
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def lay_lich_phong_van(
    opening_name_or_id: Optional[str] = None,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """Lấy lịch phỏng vấn, có thể lọc theo opening_name hoặc opening_id.
    
    Args:
        opening_name_or_id: Tên hoặc ID của vị trí tuyển dụng để lọc. Bỏ trống để lấy tất cả.
        date: Lấy lịch phỏng vấn cho 1 ngày cụ thể (YYYY-MM-DD). Nếu có tham số này, sẽ bỏ qua start_date và end_date.
        start_date: Ngày bắt đầu lọc lịch phỏng vấn (YYYY-MM-DD). Bỏ trống để lấy tất cả.
        end_date: Ngày kết thúc lọc lịch phỏng vấn (YYYY-MM-DD). Bỏ trống để lấy tất cả.
    """
    try:
        filter_date_obj = None
        
        if date:
            filter_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        
        opening_id = None
        matched_name = None
        similarity_score = None
        
        if opening_name_or_id:
            opening_id, matched_name, similarity_score = find_opening_id_by_name(
                opening_name_or_id,
                BASE_API_KEY
            )
            
            if not opening_id:
                opening_id = None
        
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
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def lay_chi_tiet_ung_vien(
    candidate_id: Optional[str] = None,
    opening_name_or_id: Optional[str] = None,
    candidate_name: Optional[str] = None
) -> dict:
    """Lấy chi tiết ứng viên. Có thể tìm bằng candidate_id, hoặc bằng opening_name_or_id + candidate_name.
    
    Args:
        candidate_id: ID của ứng viên. Bắt buộc nếu không có opening_name_or_id và candidate_name.
        opening_name_or_id: Tên hoặc ID của vị trí tuyển dụng. Bắt buộc nếu không có candidate_id.
        candidate_name: Tên ứng viên để tìm kiếm trong opening. Bắt buộc nếu không có candidate_id.
    """
    try:
        found_candidate_id = candidate_id
        opening_id = None
        opening_name_matched = None
        opening_similarity = None
        candidate_similarity = None
        
        if not found_candidate_id:
            if not opening_name_or_id or not candidate_name:
                return {
                    "success": False,
                    "error": "Phải cung cấp candidate_id, hoặc cả opening_name_or_id và candidate_name"
                }
            
            opening_id, opening_name_matched, opening_similarity = find_opening_id_by_name(
                opening_name_or_id,
                BASE_API_KEY
            )
            
            if not opening_id:
                return {
                    "success": False,
                    "error": f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Similarity score cao nhất: {opening_similarity:.2f}"
                }
            
            found_candidate_id, candidate_similarity = find_candidate_by_name_in_opening(
                candidate_name,
                opening_id,
                BASE_API_KEY,
                similarity_threshold=0.5,
                filter_stages=None
            )
            
            if not found_candidate_id:
                error_msg = f"Không tìm thấy ứng viên phù hợp với tên '{candidate_name}' trong vị trí '{opening_name_matched}'. "
                if candidate_similarity is not None:
                    error_msg += f"Candidate similarity score cao nhất: {candidate_similarity:.2f}"
                return {"success": False, "error": error_msg}
        
        candidate_data = get_candidate_details(found_candidate_id, BASE_API_KEY)
        
        cv_url = candidate_data.get('cv_url')
        cv_text = None
        if cv_url:
            cv_text = extract_text_from_cv_url_with_genai(cv_url)
            candidate_data['cv_text'] = cv_text
        
        test_results = get_test_results_from_google_sheet(found_candidate_id)
        candidate_data['test_results'] = test_results
        
        opening_name = candidate_data.get('vi_tri_ung_tuyen')
        opening_id = candidate_data.get('opening_id')
        job_description = None
        
        if opening_name or opening_id:
            if not opening_id and opening_name:
                opening_id, matched_name, similarity_score = find_opening_id_by_name(
                    opening_name,
                    BASE_API_KEY
                )
            
            if opening_id:
                jds = get_job_descriptions(BASE_API_KEY, use_cache=True)
                jd = next((jd for jd in jds if jd['id'] == opening_id), None)
                
                if not jd:
                    jds = get_job_descriptions(BASE_API_KEY, use_cache=False)
                    jd = next((jd for jd in jds if jd['id'] == opening_id), None)
                
                if jd:
                    job_description = jd['job_description']
                    candidate_data['job_description'] = job_description
        
        result = {
            "success": True,
            "candidate_id": found_candidate_id,
            "candidate_details": candidate_data
        }
        
        if opening_similarity is not None:
            result["opening_similarity_score"] = opening_similarity
            result["opening_id"] = opening_id
            result["opening_name"] = opening_name_matched
        if candidate_similarity is not None:
            result["candidate_similarity_score"] = candidate_similarity
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def lay_offer_letter_theo_ung_vien(
    candidate_id: Optional[str] = None,
    opening_name_or_id: Optional[str] = None,
    candidate_name: Optional[str] = None
) -> dict:
    """Lấy offer letter của ứng viên. Có thể tìm bằng candidate_id, hoặc bằng opening_name_or_id + candidate_name.
    
    Args:
        candidate_id: ID của ứng viên. Bắt buộc nếu không có opening_name_or_id và candidate_name.
        opening_name_or_id: Tên hoặc ID của vị trí tuyển dụng. Bắt buộc nếu không có candidate_id.
        candidate_name: Tên ứng viên để tìm kiếm trong opening. Bắt buộc nếu không có candidate_id.
    """
    try:
        found_candidate_id = candidate_id
        opening_id = None
        opening_name_matched = None
        opening_similarity = None
        candidate_similarity = None
        
        if not found_candidate_id:
            if not opening_name_or_id or not candidate_name:
                return {
                    "success": False,
                    "error": "Phải cung cấp candidate_id, hoặc cả opening_name_or_id và candidate_name"
                }
            
            opening_id, opening_name_matched, opening_similarity = find_opening_id_by_name(
                opening_name_or_id,
                BASE_API_KEY
            )
            
            if not opening_id:
                return {
                    "success": False,
                    "error": f"Không tìm thấy vị trí phù hợp với '{opening_name_or_id}'. Similarity score cao nhất: {opening_similarity:.2f}"
                }
            
            found_candidate_id, candidate_similarity = find_candidate_by_name_in_opening(
                candidate_name,
                opening_id,
                BASE_API_KEY,
                similarity_threshold=0.5,
                filter_stages=['Offered', 'Hired']
            )
            
            if not found_candidate_id:
                error_msg = f"Không tìm thấy ứng viên phù hợp với tên '{candidate_name}' trong vị trí '{opening_name_matched}'. "
                if candidate_similarity is not None:
                    error_msg += f"Candidate similarity score cao nhất: {candidate_similarity:.2f}"
                return {"success": False, "error": error_msg}
        
        candidate_data = get_candidate_details(found_candidate_id, BASE_API_KEY)
        
        candidate_name_result = candidate_data.get('ten')
        vi_tri_ung_tuyen = candidate_data.get('vi_tri_ung_tuyen')
        
        offer_letter = get_offer_letter(found_candidate_id, BASE_API_KEY)
        
        if not offer_letter:
            return {
                "success": False,
                "error": f"Không tìm thấy offer letter cho ứng viên với ID '{found_candidate_id}'"
            }
        
        result = {
            "success": True,
            "candidate_id": found_candidate_id,
            "candidate_name": candidate_name_result,
            "vi_tri_ung_tuyen": vi_tri_ung_tuyen,
            "offer_letter": offer_letter
        }
        
        if opening_similarity is not None:
            result["opening_similarity_score"] = opening_similarity
            result["opening_id"] = opening_id
            result["opening_name"] = opening_name_matched
        if candidate_similarity is not None:
            result["candidate_similarity_score"] = candidate_similarity
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run()
