"""
FastAPI Backend - Base Hiring API (CustomGPT Optimized Edition)
Feature: Extract JD, CV, Interviews, Offer Letters, Test Results
Optimization: ORJSON, Pydantic Aliases, Null filtering
"""

import os
import re
import requests
import numpy as np
import pdfplumber
from time import time
from io import BytesIO
from html import unescape
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, ConfigDict

from bs4 import BeautifulSoup
from pytz import timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# =================================================================
# 1. CONFIGURATION & APP INIT
# =================================================================

app = FastAPI(
    title="Base Hiring API - CustomGPT Optimized",
    description="API trích xuất dữ liệu tuyển dụng Base.vn, tối ưu hóa token cho LLM.",
    version="v2.1.0",
    default_response_class=ORJSONResponse  # Tối ưu tốc độ và nén JSON
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Environment Variables
BASE_API_KEY = os.getenv('BASE_API_KEY')
if not BASE_API_KEY:
    raise ValueError("BASE_API_KEY chưa được cấu hình")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY chưa được cấu hình")

GEMINI_API_KEY_DU_PHONG_STR = os.getenv('GEMINI_API_KEY_DU_PHONG', '')
GEMINI_API_KEY_DU_PHONG = [key.strip() for key in GEMINI_API_KEY_DU_PHONG_STR.split(',') if key.strip()] if GEMINI_API_KEY_DU_PHONG_STR else []

GOOGLE_SHEET_SCRIPT_URL = os.getenv('GOOGLE_SHEET_SCRIPT_URL', None)
ACCOUNT_API_KEY = os.getenv('ACCOUNT_API_KEY', None)

# Caching System
CACHE_TTL = 300  # 5 minutes
_cache = {
    'openings': {'data': None, 'timestamp': 0},
    'job_descriptions': {'data': None, 'timestamp': 0},
    'users_info': {'data': None, 'timestamp': 0}
}

# =================================================================
# 2. PYDANTIC MODELS (OPTIMIZED FOR TOKENS)
# =================================================================
# ConfigDict(populate_by_name=True) cho phép code Python dùng tên dài (dễ đọc)
# nhưng khi trả về JSON cho GPT sẽ dùng alias ngắn (tiết kiệm token).

class BaseSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

class SlimOpening(BaseSchema):
    id: str
    name: str

class SlimReview(BaseSchema):
    name: str = Field(..., alias="n")
    title: Optional[str] = Field(None, alias="t")
    content: str = Field(..., alias="c")

class SlimTest(BaseSchema):
    test_name: Optional[str] = Field(None, alias="tn")
    score: Optional[str] = Field(None, alias="s")
    link: Optional[str] = Field(None, alias="l")
    test_content: Optional[str] = Field(None, alias="tc")

class SlimCandidate(BaseSchema):
    id: str = Field(..., alias="cid")
    name: str = Field(..., alias="n")
    email: Optional[str] = Field(None, alias="e")
    phone: Optional[str] = Field(None, alias="p")
    cv_url: Optional[str] = Field(None, alias="cv")
    cv_text: Optional[str] = Field(None, alias="cv_txt")
    reviews: Optional[List[SlimReview]] = Field(None, alias="revs")
    stage_name: Optional[str] = Field(None, alias="stg")
    form_data: Optional[Dict[str, Any]] = Field(None, alias="frm")

# Response Models
class JDResponse(BaseSchema):
    found: bool
    query: Optional[str] = None
    sim: Optional[float] = None
    oid: Optional[str] = None
    oname: Optional[str] = None
    jd: Optional[str] = None
    suggestions: Optional[List[SlimOpening]] = None

class ListCandidateResponse(BaseSchema):
    oid: str
    oname: str
    sim: float
    total: int
    jd: Optional[str] = None
    candidates: List[SlimCandidate] = Field(..., alias="cands")

class SlimInterview(BaseSchema):
    id: str
    candidate_name: str = Field(..., alias="cn")
    opening_name: str = Field(..., alias="on")
    time_dt: Optional[str] = Field(None, alias="dt")

class InterviewResponse(BaseSchema):
    total: int
    interviews: List[SlimInterview] = Field(..., alias="ints")

class CandidateDetailResponse(BaseSchema):
    cid: str
    details: Dict[str, Any] # Flat dictionary
    cv_txt: Optional[str] = None
    tests: Optional[List[SlimTest]] = None
    jd: Optional[str] = None
    sim_op: Optional[float] = None
    sim_cand: Optional[float] = None

class OfferLetterResponse(BaseSchema):
    cid: str
    cname: str = Field(..., alias="n")
    position: str = Field(..., alias="pos")
    letter_text: str = Field(..., alias="txt")
    letter_url: Optional[str] = Field(None, alias="url")
    sim_op: Optional[float] = None
    sim_cand: Optional[float] = None

class TestResultResponse(BaseSchema):
    cid: str
    cname: Optional[str] = Field(None, alias="n")
    test_name: Optional[str] = Field(None, alias="tn")
    result: Optional[SlimTest] = Field(None, alias="res")
    sim_test: Optional[float] = None
    sim_cand: Optional[float] = None

# =================================================================
# 3. HELPER FUNCTIONS (LOGIC CORE)
# =================================================================

def remove_html_tags(text):
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    return unescape(text).strip()

def get_base_openings(api_key, use_cache=True):
    current_time = time()
    if use_cache and _cache['openings']['data'] is not None:
        if current_time - _cache['openings']['timestamp'] < CACHE_TTL:
            return _cache['openings']['data']
    
    try:
        url = "https://hiring.base.vn/publicapi/v2/opening/list"
        resp = requests.post(url, headers={'Content-Type': 'application/x-www-form-urlencoded'}, data={'access_token': api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        filtered = [{"id": o['id'], "name": o['name']} for o in data.get('openings', []) if o.get('status') == '10']
        if use_cache:
            _cache['openings'] = {'data': filtered, 'timestamp': current_time}
        return filtered
    except Exception:
        return []

def get_job_descriptions(api_key, use_cache=True):
    current_time = time()
    if use_cache and _cache['job_descriptions']['data'] is not None:
        if current_time - _cache['job_descriptions']['timestamp'] < CACHE_TTL:
            return _cache['job_descriptions']['data']
    
    try:
        resp = requests.post("https://hiring.base.vn/publicapi/v2/opening/list", data={'access_token': api_key}, timeout=15)
        data = resp.json()
        results = []
        for op in data.get('openings', []):
            if op.get('status') == '10':
                soup = BeautifulSoup(op.get('content', ''), "html.parser")
                text = soup.get_text()
                if len(text) >= 10:
                    results.append({"id": op['id'], "name": op['name'], "job_description": text.strip()})
        if use_cache:
            _cache['job_descriptions'] = {'data': results, 'timestamp': current_time}
        return results
    except Exception:
        return []

def get_users_info(use_cache=True):
    if not ACCOUNT_API_KEY: return {}
    current_time = time()
    if use_cache and _cache['users_info']['data'] is not None:
        if current_time - _cache['users_info']['timestamp'] < CACHE_TTL: return _cache['users_info']['data']
    try:
        resp = requests.post("https://account.base.vn/extapi/v1/users", data={'access_token': ACCOUNT_API_KEY}, timeout=10)
        users = resp.json().get('users', [])
        info = {}
        for u in users:
            username = u.get('username')
            if username:
                info[username] = {"name": u.get('name', ''), "title": "CEO" if u.get('name') == "Hoang Tran" else u.get('title', '')}
        if use_cache: _cache['users_info'] = {'data': info, 'timestamp': current_time}
        return info
    except: return {}

def process_evaluations(evaluations):
    if not evaluations: return []
    user_info_map = get_users_info()
    reviews = []
    for e in evaluations:
        if 'content' in e:
            u = e.get('username')
            ui = user_info_map.get(u, {})
            reviews.append({
                "name": ui.get('name', u or "N/A"),
                "title": ui.get('title', ''),
                "content": remove_html_tags(e.get('content', ''))
            })
    return reviews

def extract_text_from_pdf(url=None, file_bytes=None):
    pdf_file = file_bytes
    if not pdf_file and url:
        try:
            r = requests.get(url, timeout=30)
            pdf_file = BytesIO(r.content)
        except: return None
    if not pdf_file: return None
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for p in pdf.pages:
                extracted = p.extract_text()
                if extracted: text += extracted + "\n"
        return text.strip() if text else None
    except: return None

def extract_text_from_cv_url_with_genai(url):
    if not url: return None
    text = extract_text_from_pdf(url)
    if text: return text
    
    keys = [GEMINI_API_KEY] + GEMINI_API_KEY_DU_PHONG
    for api_key in keys:
        try:
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=f"{url}\nĐọc toàn bộ text trong file này")])]
            tools = [types.Tool(url_context=types.UrlContext())]
            conf = types.GenerateContentConfig(tools=tools, system_instruction=[types.Part.from_text(text="Trích xuất full text.")])
            full_text = ""
            for chunk in client.models.generate_content_stream(model="gemini-flash-lite-latest", contents=contents, config=conf):
                if chunk.text: full_text += chunk.text
            if full_text.strip(): return full_text.strip()
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower(): continue
            pass
    return None

def find_opening_id_by_name(query, api_key, threshold=0.5):
    openings = get_base_openings(api_key)
    if not openings: return None, None, 0.0
    exact = next((o for o in openings if o['id'] == query or o['name'] == query), None)
    if exact: return exact['id'], exact['name'], 1.0
    names = [o['name'] for o in openings]
    try:
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform(names + [query])
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        idx = np.argmax(sims)
        best_sim = sims[idx]
        if best_sim >= threshold:
            return openings[idx]['id'], openings[idx]['name'], float(best_sim)
        return None, None, float(best_sim)
    except: return None, None, 0.0

def find_candidate_by_name_in_opening(c_name, op_id, api_key, threshold=0.5, filter_stages=None):
    if not c_name or not op_id: return None, 0.0
    try:
        resp = requests.post("https://hiring.base.vn/publicapi/v2/candidate/list", 
                             data={'access_token': api_key, 'opening_id': op_id}, timeout=15)
        cands = resp.json().get('candidates', [])
    except: return None, 0.0

    if filter_stages:
        cands = [c for c in cands if c.get('stage_name') in filter_stages]
    if not cands: return None, 0.0
    
    exact = next((c for c in cands if c.get('name') == c_name), None)
    if exact: return exact.get('id'), 1.0

    names = [c.get('name', '') for c in cands]
    try:
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform(names + [c_name])
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        idx = np.argmax(sims)
        best_sim = sims[idx]
        if best_sim >= threshold:
            return cands[idx].get('id'), float(best_sim)
        return None, float(best_sim)
    except: return None, 0.0

def get_test_results_from_google_sheet(cid):
    if not GOOGLE_SHEET_SCRIPT_URL: return None
    try:
        resp = requests.post(GOOGLE_SHEET_SCRIPT_URL, json={'action': 'read_data', 'filters': {'candidate_id': str(cid)}}, timeout=8)
        data = resp.json().get('data', [])
        # Convert keys to English for Pydantic mapping
        results = []
        for i in data:
            results.append({
                "test_name": i.get('Tên bài test'),
                "score": i.get('Score'),
                "link": i.get('Link'),
                "test_content": i.get('test content')
            })
        return results if results else None
    except: return None

def find_test_by_name(tests, query, threshold=0.5):
    if not tests or not query: return None, 0.0
    exact = next((t for t in tests if t.get('test_name') == query), None)
    if exact: return exact, 1.0
    
    names = [t.get('test_name', '') for t in tests if t.get('test_name')]
    try:
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform(names + [query])
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        idx = np.argmax(sims)
        if sims[idx] >= threshold:
            match_name = names[idx]
            return next((t for t in tests if t.get('test_name') == match_name), None), float(sims[idx])
        return None, float(sims[idx])
    except: return None, 0.0

def download_file_to_bytes(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        return BytesIO(r.content) if r.status_code == 200 else None
    except: return None

def extract_text_doc_pdf(url, name):
    if not url: return None
    file_bytes = download_file_to_bytes(url)
    if not file_bytes: return None
    
    ext = name.lower().split('.')[-1] if '.' in name else 'pdf'
    if 'pdf' in ext: return extract_text_from_pdf(file_bytes=file_bytes)
    if 'docx' in ext and DOCX_AVAILABLE:
        try: return "\n".join([p.text for p in Document(file_bytes).paragraphs]).strip()
        except: return None
    return None

def get_offer_letter(cid, api_key):
    try:
        resp = requests.post("https://hiring.base.vn/publicapi/v2/candidate/messages", 
                             data={'access_token': api_key, 'id': cid}, timeout=15)
        msgs = resp.json().get('messages', [])
        for m in msgs:
            # Attachments
            if m.get('has_attachment'):
                for att in m.get('attachments', []):
                    url = att.get('src') or att.get('url')
                    name = att.get('name', '')
                    if url and any(x in name.lower() for x in ['.pdf', '.docx']):
                        txt = extract_text_doc_pdf(url, name)
                        if txt: return {"url": url, "name": name, "text": txt}
            # HTML Links
            if m.get('content'):
                soup = BeautifulSoup(m['content'], 'html.parser')
                for a in soup.find_all('a', href=True):
                    url = a['href']
                    name = a.get_text()
                    if any(x in url.lower() for x in ['.pdf', '.docx']):
                        txt = extract_text_doc_pdf(url, name)
                        if txt: return {"url": url, "name": name, "text": txt}
        return None
    except: return None

def get_candidate_details_full(cid, api_key):
    try:
        resp = requests.post("https://hiring.base.vn/publicapi/v2/candidate/get", data={'access_token': api_key, 'id': cid}, timeout=15)
        raw = resp.json()
    except: raise HTTPException(503, "Base API Error")
    
    if raw.get('code') != 1 or not raw.get('candidate'): raise HTTPException(404, "Not found")
    c = raw['candidate']
    
    flat = {
        'id': c.get('id'),
        'ten': c.get('name'),
        'email': c.get('email'),
        'phone': c.get('phone'),
        'opening_name': (c.get('evaluations') or [{}])[0].get('opening_export', {}).get('name', c.get('title')),
        'opening_id': (c.get('evaluations') or [{}])[0].get('opening_export', {}).get('id'),
        'stage': c.get('stage_name', c.get('status')),
        'cv_url': (c.get('cvs') or [None])[0],
        'reviews': process_evaluations(c.get('evaluations', []))
    }
    for f in c.get('fields', []) + c.get('form', []):
        if isinstance(f, dict) and 'id' in f: flat[f['id']] = f.get('value')
    return flat

# =================================================================
# 4. API ENDPOINTS
# =================================================================

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "Base Hiring API v2.1 (Optimized)"}

@app.get("/api/opening/job-description", 
         response_model=JDResponse, 
         response_model_exclude_none=True,
         operation_id="getJobDescription")
async def get_job_description(q: Optional[str] = Query(None, alias="opening_name_or_id")):
    """Lấy Job Description. Tìm theo tên hoặc ID."""
    openings = get_base_openings(BASE_API_KEY)
    if not q:
        return {"found": False, "suggestions": openings}
    
    oid, name, sim = find_opening_id_by_name(q, BASE_API_KEY)
    if not oid:
        return {"found": False, "query": q, "sim": sim, "suggestions": openings}
    
    jds = get_job_descriptions(BASE_API_KEY)
    jd_obj = next((j for j in jds if j['id'] == oid), None)
    
    if not jd_obj:
         return {"found": False, "query": q, "sim": sim, "oid": oid, "oname": name, "suggestions": openings}
         
    return {
        "found": True,
        "query": q,
        "sim": sim,
        "oid": oid,
        "oname": name,
        "jd": jd_obj['job_description']
    }

@app.get("/api/opening/{opening_name_or_id}/candidates", 
         response_model=ListCandidateResponse, 
         response_model_exclude_none=True,
         operation_id="getCandidates")
async def get_candidates(
    q: str = Path(..., alias="opening_name_or_id"),
    start: Optional[str] = Query(None, alias="start_date"),
    end: Optional[str] = Query(None, alias="end_date"),
    stage: Optional[str] = Query(None, alias="stage_name")
):
    """Lấy danh sách ứng viên theo vị trí tuyển dụng."""
    oid, name, sim = find_opening_id_by_name(q, BASE_API_KEY)
    if not oid: raise HTTPException(404, f"Không tìm thấy opening '{q}'")
    
    s_date = datetime.strptime(start, "%Y-%m-%d").date() if start else None
    e_date = datetime.strptime(end, "%Y-%m-%d").date() if end else None

    # Lấy list candidates
    try:
        payload = {'access_token': BASE_API_KEY, 'opening_id': oid}
        if s_date: payload['start_date'] = start
        if e_date: payload['end_date'] = end
        resp = requests.post("https://hiring.base.vn/publicapi/v2/candidate/list", data=payload, timeout=15)
        all_cands = resp.json().get('candidates', [])
    except: all_cands = []

    # Lọc stage
    target_cands = all_cands
    if stage:
        unique_stages = list(set([c.get('stage_name') for c in all_cands if c.get('stage_name')]))
        matched_stages = None
        if stage in unique_stages: matched_stages = [stage]
        else:
            # Cosine sim cho stage name
            try:
                vec = TfidfVectorizer()
                tfidf = vec.fit_transform(unique_stages + [stage])
                sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
                if len(sims) > 0 and np.max(sims) >= 0.3:
                    matched_stages = [unique_stages[np.argmax(sims)]]
            except: pass
        if matched_stages:
            target_cands = [c for c in all_cands if c.get('stage_name') in matched_stages]

    # Map to SlimCandidate format
    output_cands = []
    for c in target_cands:
        cv_url = (c.get('cvs') or [None])[0]
        form_d = {f['id']: f['value'] for f in c.get('form', []) if 'id' in f}
        output_cands.append({
            "id": c.get('id'),
            "name": c.get('name'),
            "email": c.get('email'),
            "phone": c.get('phone'),
            "cv_url": cv_url,
            "cv_text": extract_text_from_cv_url_with_genai(cv_url) if cv_url else None, # Lazy extraction
            "reviews": process_evaluations(c.get('evaluations', [])),
            "stage_name": c.get('stage_name'),
            "form_data": form_d
        })
    
    # Get JD for context
    jds = get_job_descriptions(BASE_API_KEY)
    jd_text = next((j['job_description'] for j in jds if j['id'] == oid), None)

    return {
        "oid": oid,
        "oname": name,
        "sim": sim,
        "total": len(output_cands),
        "jd": jd_text,
        "candidates": output_cands
    }

@app.get("/api/interviews", 
         response_model=InterviewResponse, 
         response_model_exclude_none=True,
         operation_id="getInterviews")
async def get_interviews(
    q: Optional[str] = Query(None, alias="opening_name_or_id"),
    date_str: Optional[str] = Query(None, alias="date")
):
    """Lấy lịch phỏng vấn, lọc theo ngày hoặc vị trí."""
    oid = None
    if q: oid, _, _ = find_opening_id_by_name(q, BASE_API_KEY)
    filter_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    
    try:
        resp = requests.post("https://hiring.base.vn/publicapi/v2/interview/list", data={'access_token': BASE_API_KEY}, timeout=10)
        raw = resp.json().get('interviews', [])
    except: raw = []

    filtered = []
    hcm = timezone('Asia/Ho_Chi_Minh')
    
    for i in raw:
        if oid and i.get('opening_id') != oid: continue
        
        t_iso = None
        if i.get('time'):
            dt = datetime.fromtimestamp(int(i['time']), tz=timezone('UTC')).astimezone(hcm)
            t_iso = dt.isoformat()
            if filter_date and dt.date() != filter_date: continue
        
        filtered.append({
            "id": i.get('id'),
            "candidate_name": i.get('candidate_name'),
            "opening_name": i.get('opening_name'),
            "time_dt": t_iso
        })

    return {"total": len(filtered), "interviews": filtered}

@app.get("/api/candidate", 
         response_model=CandidateDetailResponse, 
         response_model_exclude_none=True,
         operation_id="getCandidateDetail")
async def get_candidate_detail(
    cid: Optional[str] = Query(None, alias="candidate_id"),
    op_q: Optional[str] = Query(None, alias="opening_name_or_id"),
    c_name: Optional[str] = Query(None, alias="candidate_name")
):
    """Lấy chi tiết ứng viên (CV full, Test results)."""
    final_cid = cid
    sim_op = None
    sim_cand = None
    
    if not final_cid:
        if not op_q or not c_name: raise HTTPException(400, "Thiếu thông tin định danh ứng viên")
        oid, _, sim_op = find_opening_id_by_name(op_q, BASE_API_KEY)
        if not oid: raise HTTPException(404, "Opening not found")
        final_cid, sim_cand = find_candidate_by_name_in_opening(c_name, oid, BASE_API_KEY)
        if not final_cid: raise HTTPException(404, "Candidate not found")

    details = get_candidate_details_full(final_cid, BASE_API_KEY)
    
    # Full Extract CV
    cv_txt = None
    if details.get('cv_url'):
        cv_txt = extract_text_from_cv_url_with_genai(details['cv_url'])
    
    # Get Tests
    tests = get_test_results_from_google_sheet(final_cid)
    
    # Get JD
    jd = None
    if details.get('opening_id'):
        jds = get_job_descriptions(BASE_API_KEY)
        jd = next((j['job_description'] for j in jds if j['id'] == details['opening_id']), None)

    return {
        "cid": final_cid,
        "details": details,
        "cv_txt": cv_txt,
        "tests": tests,
        "jd": jd,
        "sim_op": sim_op,
        "sim_cand": sim_cand
    }

@app.get("/api/offer-letter",
         response_model=OfferLetterResponse,
         response_model_exclude_none=True,
         operation_id="getOfferLetter")
async def get_offer_letter_endpoint(
    cid: Optional[str] = Query(None, alias="candidate_id"),
    op_q: Optional[str] = Query(None, alias="opening_name_or_id"),
    c_name: Optional[str] = Query(None, alias="candidate_name")
):
    """Tìm và trích xuất nội dung Offer Letter."""
    final_cid = cid
    sim_op = None
    sim_cand = None

    if not final_cid:
        if not op_q or not c_name: raise HTTPException(400, "Thiếu thông tin")
        oid, _, sim_op = find_opening_id_by_name(op_q, BASE_API_KEY)
        if not oid: raise HTTPException(404, "Opening not found")
        final_cid, sim_cand = find_candidate_by_name_in_opening(c_name, oid, BASE_API_KEY, filter_stages=['Offered', 'Hired'])
        if not final_cid: raise HTTPException(404, "Candidate not found in Offered/Hired stage")

    details = get_candidate_details_full(final_cid, BASE_API_KEY)
    offer = get_offer_letter(final_cid, BASE_API_KEY)
    
    if not offer: raise HTTPException(404, "No offer letter found")

    return {
        "cid": final_cid,
        "cname": details.get('ten', ''),
        "position": details.get('opening_name', ''),
        "letter_text": offer.get('text'),
        "letter_url": offer.get('url'),
        "sim_op": sim_op,
        "sim_cand": sim_cand
    }

@app.get("/api/test-result",
         response_model=TestResultResponse,
         response_model_exclude_none=True,
         operation_id="getTestResult")
async def get_test_result_endpoint(
    t_name: str = Query(..., alias="test_name"),
    cid: Optional[str] = Query(None, alias="candidate_id"),
    op_q: Optional[str] = Query(None, alias="opening_name_or_id"),
    c_name: Optional[str] = Query(None, alias="candidate_name")
):
    """Tìm kết quả bài test cụ thể."""
    final_cid = cid
    
    # Logic tìm Candidate ID nếu thiếu (tương tự trên nhưng thêm logic tìm trong Sheet)
    # Để đơn giản hóa cho bản optimized, ta reuse logic tìm trong Base trước
    if not final_cid:
        if not op_q or not c_name: raise HTTPException(400, "Thiếu thông tin")
        oid, _, _ = find_opening_id_by_name(op_q, BASE_API_KEY)
        if oid:
             final_cid, _ = find_candidate_by_name_in_opening(c_name, oid, BASE_API_KEY)
        
        # Fallback: Nếu không thấy trong Base, user có thể implement tìm fuzzy trong Sheet
        if not final_cid: raise HTTPException(404, "Candidate not found")

    tests = get_test_results_from_google_sheet(final_cid)
    if not tests: raise HTTPException(404, "No tests found")

    found_test, sim = find_test_by_name(tests, t_name)
    if not found_test: raise HTTPException(404, "Test not found")

    # Lấy tên candidate từ list test (nếu có) hoặc fallback
    return {
        "cid": final_cid,
        "test_name": t_name,
        "result": found_test,
        "sim_test": sim
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
