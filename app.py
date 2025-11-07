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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import pdfplumber
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google import genai
from google.genai import types
from pytz import timezone
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time as time_module

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

# Base.vn Login Credentials (for test data extraction)
BASE_EMAIL = os.getenv('BASE_EMAIL')
BASE_PASSWORD = os.getenv('BASE_PASSWORD')

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
            
            review = extract_message(candidate.get('evaluations', []))
            
            form_data = {}
            if 'form' in candidate and isinstance(candidate['form'], list):
                for item in candidate['form']:
                    if isinstance(item, dict) and 'id' in item and 'value' in item:
                        form_data[item['id']] = item['value']
            
            # Lấy test data
            test_data = get_candidate_test_data(opening_id, candidate.get('id'))
            
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
                "opening_id": opening_id,
                "stage_id": candidate.get('stage_id'),
                "stage_name": candidate.get('stage_name'),
                "test_data": test_data
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

def get_candidate_test_data(opening_id, candidate_id):
    """Lấy dữ liệu bài test của ứng viên từ trang Base.vn bằng Selenium"""
    if not BASE_EMAIL or not BASE_PASSWORD:
        return None
    
    driver = None
    try:
        # Khởi tạo WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')

        # BỎ 'service=Service(ChromeDriverManager().install())'
        # Khi cài Chrome/ChromeDriver trong Docker, Selenium sẽ tự động tìm trong PATH
        driver = webdriver.Chrome(options=chrome_options)
        
        # Tạo URL đăng nhập với return URL
        target_url = f"https://hiring.base.vn/opening/{opening_id}?candidate={candidate_id}"
        login_url = f"https://account.base.vn/a/login?app=hiring&return=opening%2F{opening_id}%3Fcandidate%3D{candidate_id}"
        
        # Đăng nhập
        driver.get(login_url)
        wait = WebDriverWait(driver, 15)
        
        # Điền email
        email_input = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        email_input.clear()
        email_input.send_keys(BASE_EMAIL)
        time_module.sleep(1)
        
        # Điền password
        password_input = None
        password_selectors = [
            (By.NAME, "login-password"),
            (By.ID, "login-password"),
            (By.ID, "password"),
            (By.NAME, "password"),
            (By.CSS_SELECTOR, "input[type='password']"),
        ]
        
        for selector_type, selector_value in password_selectors:
            try:
                password_input = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                break
            except:
                continue
        
        if password_input is None:
            return None
        
        password_input.clear()
        password_input.send_keys(BASE_PASSWORD)
        
        # Nhấn nút đăng nhập
        login_button = None
        button_selectors = [
            (By.XPATH, "//button[@type='submit']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//button[contains(text(), 'Đăng nhập')]"),
            (By.XPATH, "//button[contains(text(), 'Login')]"),
        ]
        
        for selector_type, selector_value in button_selectors:
            try:
                login_button = driver.find_element(selector_type, selector_value)
                if login_button.is_displayed():
                    login_button.click()
                    break
            except:
                continue
        
        if login_button is None:
            password_input.send_keys(Keys.RETURN)
        
        # Đợi đăng nhập hoàn tất
        time_module.sleep(3)
        
        # Điều hướng đến trang mục tiêu nếu chưa ở đó
        current_url = driver.current_url
        if target_url not in current_url:
            driver.get(target_url)
        
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time_module.sleep(2)
        
        # Parse HTML để lấy dữ liệu test
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Tìm phần "Kết quả bài kiểm tra"
        test_result_section = None
        title_elements = soup.find_all(string=lambda text: text and "Kết quả bài kiểm tra" in text.strip())
        
        if title_elements:
            for title_elem in title_elements:
                parent = title_elem.find_parent()
                if parent:
                    section = parent.find_parent(class_='events')
                    if section:
                        test_result_section = section
                        break
                    if parent.find_parent(class_='body'):
                        test_result_section = parent.find_parent(class_='body')
                        break
        
        if test_result_section is None:
            events_sections = soup.find_all('div', class_='events')
            for section in events_sections:
                title_div = section.find('div', class_='title')
                if title_div and "Kết quả bài kiểm tra" in title_div.get_text():
                    test_result_section = section
                    break
        
        if not test_result_section:
            return None
        
        # Trích xuất dữ liệu từ bảng test-results
        test_table = test_result_section.find('table', class_='test-results')
        if not test_table:
            return None
        
        test_rows = test_table.find_all('tr', class_='test-result')
        test_data = []
        
        for row in test_rows:
            cells = row.find_all('td')
            if len(cells) >= 5:
                name_cell = cells[0].find('div', class_='name')
                test_name = name_cell.get_text(strip=True) if name_cell else ""
                
                score_cell = cells[1].find('div', class_='score')
                score = score_cell.get_text(strip=True) if score_cell else ""
                
                time_cell = cells[2].find('div', class_='time')
                test_time = time_cell.get_text(strip=True) if time_cell else ""
                
                link_cell = cells[4].find('div', class_='actions')
                link = ""
                if link_cell:
                    link_tag = link_cell.find('a')
                    if link_tag:
                        link = link_tag.get('href', '')
                
                test_data.append({
                    'test_name': test_name,
                    'score': score,
                    'time': test_time,
                    'link': link,
                    'test_content': None  # Sẽ được điền sau nếu cần
                })
        
        # Lấy nội dung chi tiết từ các link (tùy chọn, có thể bỏ qua để tăng tốc)
        # for test_item in test_data:
        #     if test_item['link']:
        #         # Có thể implement logic lấy test_content từ link ở đây nếu cần
        #         pass
        
        return test_data if test_data else None
        
    except Exception as e:
        # Log lỗi nhưng không raise để không làm gián đoạn flow chính
        print(f"Lỗi khi lấy test data cho candidate {candidate_id}: {str(e)}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

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
    
    return refined_data

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
        
        # Lấy test data (chỉ khi có opening_id)
        test_data = None
        if opening_id:
            test_data = get_candidate_test_data(opening_id, candidate_id)
        candidate_data['test_data'] = test_data
        
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
