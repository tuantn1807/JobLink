# -*- coding: utf-8 -*-
"""Build and export a single unified, deduplicated CSV dataset for the NCKH job matching system.

Combines:
1. Local Excel datasets: jobs_vietnamworks.xlsx, db_base.xlsx
2. Hugging Face dinhieufam/VietJobs (downloads/VietJobs.csv)
3. Kaggle quocnguyenx43/vietnamese-job-description-dataset (downloads/vn_job_desc.zip -> raw_data.csv)

Output:
- data/jobs_dataset.csv: Single deduplicated CSV dataset across all 4 sources.
"""

import ast
import csv
import hashlib
import json
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

from config import COL, VN_CITY_ALIASES

csv.field_size_limit(10**7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
DATA_DIR = os.path.join(BASE_DIR, "data")

STANDARD_COLS = [
    COL["job_id"],
    COL["job_url"],
    COL["job_title"],
    COL["company"],
    COL["location"],
    COL["location_detail"],
    COL["requirements"],
    COL["salary"],
    COL["experience"],
    COL["job_desc"],
    COL["job_type"],
    COL["benefit"],
]

EXTRA_COLS = [
    "Industry",
    "Technical_Skills",
    "Soft_Skills",
    "Qualifications",
    "Source",
]

ALL_COLS = STANDARD_COLS + EXTRA_COLS

_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXP_EXTRACT_RE = re.compile(
    r"(\d+\s*[-–~]?\s*\d*\s*(?:năm|nam|year|years)\b|không yêu cầu(?:\s+kinh nghiệm)?|chưa có kinh nghiệm|dưới\s*1\s*năm)",
    flags=re.IGNORECASE,
)
_SAL_EXTRACT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?\s*[-–~]\s*\d+(?:[.,]\d+)?\s*(?:triệu|tr|usd|\$|m)|up to\s*\d+(?:[.,]\d+)?\s*(?:triệu|tr|usd|\$))",
    flags=re.IGNORECASE,
)

# Extended city aliases for inference on description-only datasets
_EXTRA_CITY_MAP = {
    **VN_CITY_ALIASES,
    "ho chi minh": "Hồ Chí Minh",
    "tp hcm": "Hồ Chí Minh",
    "tphcm": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh",
    "sai gon": "Hồ Chí Minh",
    "ha noi": "Hà Nội",
    "hanoi": "Hà Nội",
    "da nang": "Đà Nẵng",
    "danang": "Đà Nẵng",
    "binh duong": "Bình Dương",
    "dong nai": "Đồng Nai",
    "hai phong": "Hải Phòng",
    "can tho": "Cần Thơ",
    "bac ninh": "Bắc Ninh",
    "hung yen": "Hưng Yên",
    "long an": "Long An",
    "khanh hoa": "Khánh Hòa",
    "nha trang": "Khánh Hòa",
    "vung tau": "Bà Rịa - Vũng Tàu",
}


def norm_ascii(text: str) -> str:
    """Normalize Vietnamese text to ASCII lowercase for matching and deduplication."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", str(text).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def clean_text(val) -> str:
    """Sanitize text by removing illegal XML characters and normalizing escapes."""
    if val is None:
        return ""
    s = str(val)
    if s.lower() == "nan":
        return ""
    s = s.replace("\\u003c", "<").replace("\\u003e", ">").replace("\\t", " ")
    s = _XML_ILLEGAL_RE.sub("", s)
    return s.strip()


def parse_list_field(raw: str) -> list:
    """Safely parse stringified Python list fields from VietJobs.csv."""
    s = clean_text(raw)
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [clean_text(x) for x in parsed if clean_text(x)]
        except Exception:
            pass
    return [s]


def format_list_text(raw: str) -> str:
    """Convert a stringified list field into a comma-separated string."""
    items = parse_list_field(raw)
    return ", ".join(items)


def col_letter_to_idx(ref: str) -> int:
    """Convert Excel cell reference (e.g. 'AB12') to zero-based column index."""
    m = re.match(r"([A-Z]+)", ref)
    if not m:
        return 0
    idx = 0
    for ch in m.group(1):
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_rows(filepath: str) -> list:
    """Read an .xlsx file into a list of dicts using standard library XML parsing."""
    with zipfile.ZipFile(filepath) as z:
        ss = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                ss.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in z.namelist():
            sheets = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            sheet_name = sheets[0]

        sheet = ET.fromstring(z.read(sheet_name))
        raw_rows = []
        for r in sheet.iter():
            if not r.tag.endswith("}row"):
                continue
            row_map = {}
            for idx_fallback, c in enumerate(r):
                ref = c.attrib.get("r", "")
                c_idx = col_letter_to_idx(ref) if ref else idx_fallback
                t = c.attrib.get("t")
                if t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter() if x.tag.endswith("}t"))
                else:
                    v = next((x.text for x in c if x.tag.endswith("}v")), None)
                    if t == "s" and v is not None:
                        val = ss[int(v)]
                    else:
                        val = v
                row_map[c_idx] = val
            raw_rows.append(row_map)

    if not raw_rows:
        return []

    header_map = raw_rows[0]
    records = []
    for r in raw_rows[1:]:
        rec = {}
        for c_idx, col_name in header_map.items():
            if col_name:
                rec[str(col_name).strip()] = clean_text(r.get(c_idx, ""))
        records.append(rec)
    return records


def infer_city(text: str) -> str:
    """Infer Vietnamese city name from text when location column is absent."""
    t = norm_ascii(text)
    for alias, city_name in _EXTRA_CITY_MAP.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", t):
            return city_name
    return "Unknown"


def infer_experience(text: str) -> str:
    """Extract experience phrase from requirements text."""
    m = _EXP_EXTRACT_RE.search(text or "")
    return clean_text(m.group(1)) if m else ""


def infer_salary(text: str) -> str:
    """Extract salary phrase from description text if available."""
    m = _SAL_EXTRACT_RE.search(text or "")
    return clean_text(m.group(1)) if m else "Thỏa thuận"


def dedup_fingerprint(row: dict) -> str:
    """Compute a normalized text fingerprint for cross-source deduplication."""
    desc = norm_ascii(row.get(COL["job_desc"], ""))[:220]
    req = norm_ascii(row.get(COL["requirements"], ""))[:220]
    title = norm_ascii(row.get(COL["job_title"], ""))
    # Use description + requirement core so anonymous/titled copies still deduplicate
    base = f"{desc}|{req}" if (desc or req) else title
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def load_local_vietnamworks() -> list:
    """Load jobs_vietnamworks.xlsx mapped to unified schema."""
    path = os.path.join(BASE_DIR, "jobs_vietnamworks.xlsx")
    if not os.path.exists(path):
        return []
    rows = read_xlsx_rows(path)
    out = []
    for i, r in enumerate(rows):
        title = r.get("Title", "")
        if not title:
            continue
        out.append({
            COL["job_id"]: f"vnw_{r.get('JobID') or i}",
            COL["job_url"]: r.get("URL_Job", ""),
            COL["job_title"]: title,
            COL["company"]: r.get("Name company", "") or "Unknown",
            COL["location"]: r.get("Job Address", "") or "Unknown",
            COL["location_detail"]: r.get("Job Address detail", ""),
            COL["requirements"]: r.get("Job Requirements", ""),
            COL["salary"]: r.get("Salary", "") or "Thỏa thuận",
            COL["experience"]: r.get("Experience", ""),
            COL["job_desc"]: r.get("Job description", ""),
            COL["job_type"]: r.get("Job type", ""),
            COL["benefit"]: r.get("benefit", ""),
            "Industry": "",
            "Technical_Skills": "",
            "Soft_Skills": "",
            "Qualifications": "",
            "Source": "jobs_vietnamworks",
        })
    return out


def load_local_db_base() -> list:
    """Load non-empty rows from db_base.xlsx and normalize Address detail column."""
    path = os.path.join(BASE_DIR, "db_base.xlsx")
    if not os.path.exists(path):
        return []
    rows = read_xlsx_rows(path)
    out = []
    for i, r in enumerate(rows):
        title = r.get("Title", "")
        if not title:
            continue
        out.append({
            COL["job_id"]: f"base_{r.get('JobID') or i}",
            COL["job_url"]: r.get("URL_Job", ""),
            COL["job_title"]: title,
            COL["company"]: r.get("Name company", "") or "Unknown",
            COL["location"]: r.get("Job Address", "") or "Unknown",
            COL["location_detail"]: r.get("Job Address detail") or r.get("Address detail", ""),
            COL["requirements"]: r.get("Job Requirements", ""),
            COL["salary"]: r.get("Salary", "") or "Thỏa thuận",
            COL["experience"]: r.get("Experience", ""),
            COL["job_desc"]: r.get("Job description", ""),
            COL["job_type"]: r.get("Job type", ""),
            COL["benefit"]: r.get("benefit", ""),
            "Industry": "",
            "Technical_Skills": "",
            "Soft_Skills": "",
            "Qualifications": "",
            "Source": "db_base",
        })
    return out


def load_vietjobs() -> list:
    """Load downloads/VietJobs.csv mapped to unified schema."""
    path = os.path.join(DOWNLOADS_DIR, "VietJobs.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            title = clean_text(r.get("job_title", ""))
            if not title:
                continue
            tech_list = parse_list_field(r.get("technical_skills", ""))
            soft_list = parse_list_field(r.get("soft_skills", ""))
            qual_list = parse_list_field(r.get("qualifications", ""))
            lang_list = parse_list_field(r.get("languages_required", ""))

            req_text = clean_text(r.get("requirements_text", ""))
            extra_skill_line = ", ".join(tech_list + lang_list)
            if extra_skill_line and extra_skill_line.lower() not in req_text.lower():
                req_combined = f"{req_text}\nKỹ năng: {extra_skill_line}".strip()
            else:
                req_combined = req_text

            sal = clean_text(r.get("salary", ""))
            if not sal:
                smin, smax = clean_text(r.get("salary_min", "")), clean_text(r.get("salary_max", ""))
                sal = f"{smin} - {smax} triệu" if (smin and smax) else "Thỏa thuận"

            loc = clean_text(r.get("location", "")).title() or "Unknown"

            out.append({
                COL["job_id"]: f"vj_{i}",
                COL["job_url"]: "",
                COL["job_title"]: title,
                COL["company"]: "Unknown",
                COL["location"]: loc,
                COL["location_detail"]: loc,
                COL["requirements"]: req_combined,
                COL["salary"]: sal,
                COL["experience"]: clean_text(r.get("experience_required", "")),
                COL["job_desc"]: clean_text(r.get("description", "")),
                COL["job_type"]: clean_text(r.get("contract_type", "")),
                COL["benefit"]: format_list_text(r.get("benefits", "")),
                "Industry": clean_text(r.get("category", "")),
                "Technical_Skills": ", ".join(tech_list),
                "Soft_Skills": ", ".join(soft_list),
                "Qualifications": ", ".join(qual_list),
                "Source": "VietJobs",
            })
    return out


def load_kaggle_vn_job_desc() -> list:
    """Load downloads/vn_job_desc.zip (raw_data.csv) mapped with inferred metadata."""
    path = os.path.join(DOWNLOADS_DIR, "vn_job_desc.zip")
    if not os.path.exists(path):
        return []
    import io

    out = []
    with zipfile.ZipFile(path) as z:
        with z.open("raw_data.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            for i, r in enumerate(reader):
                desc = clean_text(r.get("description", ""))
                req = clean_text(r.get("requirements", ""))
                ind = clean_text(r.get("mapped_industry", ""))
                if not desc and not req:
                    continue

                primary_ind = ind.split(",")[0].strip() if ind else "Khác"
                title = f"Chuyên viên {primary_ind}" if primary_ind else "Nhân viên"
                city = infer_city(f"{desc[:400]} {req[:300]}")

                out.append({
                    COL["job_id"]: f"kgl_{i}",
                    COL["job_url"]: "",
                    COL["job_title"]: title,
                    COL["company"]: "Unknown",
                    COL["location"]: city,
                    COL["location_detail"]: city if city != "Unknown" else "",
                    COL["requirements"]: req,
                    COL["salary"]: infer_salary(desc),
                    COL["experience"]: infer_experience(req),
                    COL["job_desc"]: desc,
                    COL["job_type"]: "",
                    COL["benefit"]: "",
                    "Industry": ind,
                    "Technical_Skills": "",
                    "Soft_Skills": "",
                    "Qualifications": "",
                    "Source": "kaggle_vn_job_desc",
                })
    return out


def write_csv(filepath: str, columns: list, rows: list):
    """Write rows to UTF-8 CSV."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    sources = [
        ("jobs_vietnamworks", load_local_vietnamworks()),
        ("db_base", load_local_db_base()),
        ("VietJobs", load_vietjobs()),
        ("kaggle_vn_job_desc", load_kaggle_vn_job_desc()),
    ]

    seen_urls = set()
    seen_fps = set()
    merged_full = []
    dedup_stats = {}

    for src_name, src_rows in sources:
        kept = 0
        dropped = 0
        for r in src_rows:
            url = r.get(COL["job_url"], "").strip()
            fp = dedup_fingerprint(r)
            if (url and url in seen_urls) or fp in seen_fps:
                dropped += 1
                continue
            if url:
                seen_urls.add(url)
            seen_fps.add(fp)
            merged_full.append(r)
            kept += 1
        dedup_stats[src_name] = {"raw": len(src_rows), "kept": kept, "dedup_dropped": dropped}

    output_csv = os.path.join(DATA_DIR, "jobs_dataset.csv")
    write_csv(output_csv, ALL_COLS, merged_full)

    print("Export complete:")
    print(f"  - {output_csv}: {len(merged_full)} rows")
    print("Source breakdown:", json.dumps(dedup_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

