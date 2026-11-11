# -*- coding: utf-8 -*-
"""LLM-enhanced HR Knowledge Graph extraction and semantic enrichment.

Supports:
1. External/local LLMs when configured via environment variables:
   - GEMINI_API_KEY (and optional GEMINI_MODEL, default 'gemini-2.5-flash')
   - OPENAI_API_KEY (and optional OPENAI_BASE_URL, OPENAI_MODEL)
   - OLLAMA_MODEL (e.g. 'qwen2.5:7b', using OLLAMA_BASE_URL default http://localhost:11434)
2. Deterministic local semantic fallback when no LLM provider is active.
"""

import hashlib
import json
import os
import re
import requests

from config import LLM_CACHE_PATH, SKILL_LEXICON
from skill_extraction import extract_skills_probabilistic
from text_processing import (
    exp_bucket,
    infer_role_canonical,
    norm_text,
    parse_location_city_detail,
    parse_year_range,
)

_QUAL_PATTERNS = [
    (re.compile(r"\b(dai hoc|bachelor|cu nhan|ky su)\b"), "Đại học"),
    (re.compile(r"\b(cao dang|college|associate)\b"), "Cao đẳng"),
    (re.compile(r"\b(thac si|master|mba)\b"), "Thạc sĩ"),
    (re.compile(r"\b(tien si|phd|doctorate)\b"), "Tiến sĩ"),
    (re.compile(r"\b(cpa|acca|cfa)\b"), "Chứng chỉ Kế toán/Tài chính"),
    (re.compile(r"\b(ielts|toeic)\b"), "Chứng chỉ Tiếng Anh"),
]

_INDUSTRY_KEYWORDS = {
    "Công nghệ thông tin": ["python", "java", "javascript", "react", "nodejs", "docker", "kubernetes", "aws", "software", "developer", "backend", "frontend"],
    "Tài chính - Kế toán": ["ke toan", "kiem toan", "tax", "vat", "misa", "sap", "financial", "audit", "bookkeeping", "ngan hang"],
    "Marketing - Truyền thông": ["marketing", "seo", "content", "social", "photoshop", "illustrator", "canva", "quang cao"],
    "Kinh doanh - Bán hàng": ["sales", "kinh doanh", "khach hang", "b2b", "b2c", "doanh so"],
    "Thiết kế - Sáng tạo": ["design", "thiet ke", "photoshop", "illustrator", "autocad", "ui/ux"],
    "Nhân sự - Hành chính - Pháp chế": ["nhan su", "hr", "tuyen dung", "phap che", "luat", "hanh chinh"],
    "Kỹ thuật - Sản xuất": ["bao tri", "co khi", "dien", "autocad", "van hanh", "san xuat"],
}

_INDUSTRY_PATTERNS = {
    ind: [re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)") for kw in kws]
    for ind, kws in _INDUSTRY_KEYWORDS.items()
}


def _load_cache() -> dict:
    if os.path.exists(LLM_CACHE_PATH):
        try:
            with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(LLM_CACHE_PATH) or ".", exist_ok=True)
    try:
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _extract_json_block(raw_text: str) -> dict:
    """Parse JSON object from LLM text response."""
    if not raw_text:
        return {}
    m = re.search(r"\{[\s\S]*\}", raw_text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _call_llm_text(prompt: str, timeout: int = 20) -> tuple[str, str]:
    """Call configured LLM provider and return (text_response, provider_name)."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        try:
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json()
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                return txt, f"gemini:{model}"
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json()
                return data["choices"][0]["message"]["content"], f"openai:{model}"
        except Exception:
            pass

    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            if resp.ok:
                return resp.json().get("response", ""), f"ollama:{ollama_model}"
        except Exception:
            pass

    return "", "none"


def infer_industry_from_text(text: str) -> str:
    """Infer primary industry label from text cues."""
    t = norm_text(text)
    best_ind, best_hits = "Khác", 0
    for ind, pats in _INDUSTRY_PATTERNS.items():
        hits = sum(1 for p in pats if p.search(t))
        if hits > best_hits:
            best_hits = hits
            best_ind = ind
    return best_ind


def extract_qualifications_from_text(text: str) -> list[str]:
    """Extract normalized qualification nodes from text."""
    t = norm_text(text)
    found = []
    for pat, label in _QUAL_PATTERNS:
        if pat.search(t) and label not in found:
            found.append(label)
    return found


def enrich_hr_document(text: str, doc_type: str = "cv", use_llm: bool = True) -> dict:
    """Extract structured HR Knowledge Graph entities from a CV or JD text.

    Returns dict with keys:
      canonical_role, industry, city, city_detail, exp_bucket,
      skills (dict[str, float]), qualifications (list[str]), provider (str)
    """
    raw = str(text or "")
    key = hashlib.md5(f"{doc_type}:{raw}".encode("utf-8")).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    # Always compute deterministic baseline signals first
    base_skills, _ = extract_skills_probabilistic(raw)
    city, detail = parse_location_city_detail(raw)
    role_can = infer_role_canonical(raw)
    emin, emax, _ = parse_year_range(raw)
    ebucket = exp_bucket(emin, emax) if emin is not None else "Exp_Unknown"
    industry = infer_industry_from_text(raw)
    quals = extract_qualifications_from_text(raw)
    provider = "local_semantic_fallback"

    if use_llm and raw.strip():
        canon_list = ", ".join(SKILL_LEXICON.keys())
        prompt = (
            f"Bạn là chuyên gia phân tích HR Knowledge Graph. Hãy trích xuất JSON từ {doc_type.upper()} sau.\n"
            f"Danh sách kỹ năng chuẩn ưu tiên ánh xạ: [{canon_list}].\n"
            "Trả về DUY NHẤT 1 JSON object có các khóa:\n"
            '{"canonical_role": "...", "industry": "...", "city": "...", '
            '"exp_years": <số hoặc null>, "skills": {"TênKỹNăng": <xác suất 0.1-0.95>}, '
            '"qualifications": ["..."]}\n\n'
            f"Nội dung:\n{raw[:4000]}"
        )
        llm_raw, used_provider = _call_llm_text(prompt)
        parsed = _extract_json_block(llm_raw)
        if parsed:
            provider = f"llm:{used_provider}"
            if parsed.get("canonical_role"):
                role_can = str(parsed["canonical_role"])
            if parsed.get("industry"):
                industry = str(parsed["industry"])
            if parsed.get("city") and city == "Unknown":
                city = str(parsed["city"])
            if isinstance(parsed.get("exp_years"), (int, float)) and ebucket == "Exp_Unknown":
                y = float(parsed["exp_years"])
                ebucket = exp_bucket(y, y)
            if isinstance(parsed.get("skills"), dict):
                for sk, prob in parsed["skills"].items():
                    try:
                        p = min(0.95, max(0.1, float(prob)))
                    except Exception:
                        p = 0.6
                    base_skills[str(sk)] = round(max(base_skills.get(str(sk), 0.0), p), 3)
            if isinstance(parsed.get("qualifications"), list):
                for q in parsed["qualifications"]:
                    qs = str(q).strip()
                    if qs and qs not in quals:
                        quals.append(qs)

    result = {
        "canonical_role": role_can,
        "industry": industry,
        "city": city,
        "city_detail": detail,
        "exp_bucket": ebucket,
        "skills": base_skills,
        "qualifications": quals,
        "provider": provider,
    }
    cache[key] = result
    _save_cache(cache)
    return result


def generate_interview_reply(user_msg: str, user_skills: dict | None = None, top_jobs: list | None = None) -> str:
    """Generate contextual AI interview response using LLM if available, else smart fallback."""
    skills_str = ", ".join(list((user_skills or {}).keys())[:8]) or "chưa xác định"
    top_role = top_jobs[0]["title"] if top_jobs else "vị trí ứng tuyển"

    prompt = (
        f"Bạn là nhà tuyển dụng đang phỏng vấn ứng viên cho vị trí: {top_role}.\n"
        f"Kỹ năng nổi bật từ CV của ứng viên: {skills_str}.\n"
        f"Ứng viên vừa nói: \"{user_msg}\"\n"
        "Hãy phản hồi ngắn gọn (2-3 câu, tiếng Việt), nhận xét câu trả lời và đặt 1 câu hỏi phỏng vấn chuyên sâu tiếp theo."
    )
    llm_reply, _ = _call_llm_text(prompt, timeout=15)
    if llm_reply:
        return llm_reply.strip()

    msg = (user_msg or "").lower()
    if any(k in msg for k in ["intro", "myself", "giới thiệu", "tên", "kinh nghiệm"]):
        return f"Cảm ơn bạn. Với nền tảng kỹ năng ({skills_str}), bạn có thể chia sẻ một dự án thực tế mà bạn tự hào nhất cho vị trí {top_role} không?"
    if any(k in msg for k in ["python", "sql", "excel", "sap", "misa", "react", "tax", "dự án", "project"]):
        return "Rất tốt! Khi gặp bài toán khó hoặc lỗi phát sinh trong quá trình làm việc với công cụ đó, bạn thường xử lý và kiểm chứng kết quả ra sao?"
    if any(k in msg for k in ["bug", "lỗi", "khó khăn", "deadline", "áp lực"]):
        return "Tư duy giải quyết vấn đề rất quan trọng. Bạn còn câu hỏi nào dành cho đội ngũ hoặc định hướng phát triển của vị trí này không?"
    return f"Điểm này khá thú vị. Bạn có thể nói rõ hơn về cách bạn áp dụng kỹ năng ({skills_str}) vào mục tiêu công việc sắp tới không?"

