# -*- coding: utf-8 -*-
"""Configuration settings for NCKH job matching system"""

# USER–JOB MATCHING
TOPK_USER_JOB = 3
W_SKILL = 0.78
W_LOC = 0.22

SIM_THRESHOLD = 0.45
CANDIDATES_TOP = 15
TOPK_SIMILAR = 3

# DRAW
FOCUS_ONLY = True
MAX_SKILLS_DRAW = 10
EDGE_LABEL_MODE = "important"
SHOW_EDGE_SCORES = True
CENTER_MODE = "user"
CENTER_JOB_ID = None
RANDOM_SEED = 42

# CV PDF OCR settings
OCR_LANG = "vie+eng"
OCR_DPI = 300
OCR_MAX_PAGES = 99
FORCE_OCR = True

# Prob skill
MIN_KEEP_PROB = 0.08

# SKILL LEXICON
SKILL_LEXICON = {
    "Python":        ["python", "py", "python3", "python 3"],
    "Java":          ["java"],
    "JavaScript":    ["javascript", "js", "ecmascript"],
    "TypeScript":    ["typescript", "ts"],
    "React":         ["react", "reactjs", "react.js"],
    "NodeJS":        ["nodejs", "node.js", "node"],
    "SQL":           ["sql", "mysql", "postgresql", "postgres", "mssql", "sql server"],
    "Docker":        ["docker"],
    "Kubernetes":    ["kubernetes", "k8s"],
    "AWS":           ["aws", "amazon web services"],
    "Git":           ["git", "github"],
    "Excel":         ["excel", "pivot", "vlookup", "power query"],
    "SAP":           ["sap"],
    "MISA":          ["misa"],
    "QuickBooks":    ["quickbooks"],
    "VAT":           ["vat", "gtgt"],
    "Invoice":       ["hoa don", "invoice", "e-invoice", "einvoice", "e invoice"],
    "Tax":           ["thue", "tax"],
    "Audit":         ["kiem toan", "audit"],
    "AR/AP":         ["cong no", "ar", "ap", "accounts payable", "accounts receivable"],
    "Bookkeeping":   ["so sach", "bookkeeping"],
    "Financial Reporting": ["bao cao tai chinh", "financial reporting", "financial statement", "financial statements"],
    "Tax Finalization": ["quyet toan", "tax finalization"],
    "HTML/CSS":      ["html", "css", "html5", "css3", "tailwind"],
    "C#/.NET":       ["c#", ".net", "dotnet", "asp.net"],
    "PHP":           ["php", "laravel", "symfony"],
    "Spring":        ["spring", "spring boot", "springboot"],
    "Flutter":       ["flutter", "dart"],
    "Machine Learning": ["machine learning", "deep learning", "nlp", "computer vision", "ai"],
    "PyTorch":       ["pytorch"],
    "TensorFlow":    ["tensorflow", "keras"],
    "Data Analysis": ["data analysis", "phan tich du lieu", "pandas", "numpy"],
    "Power BI":      ["power bi", "powerbi"],
    "Tableau":       ["tableau"],
    "ERP":           ["erp", "odoo", "oracle"],
    "IFRS":          ["ifrs", "vas"],
    "Word":          ["ms word", "microsoft word", "word"],
    "PowerPoint":    ["powerpoint", "power point"],
    "Office Suite":  ["tin hoc van phong", "microsoft office", "ms office"],
    "Photoshop":     ["photoshop", "adobe photoshop"],
    "Illustrator":   ["illustrator", "adobe illustrator"],
    "Canva":         ["canva"],
    "AutoCAD":       ["autocad", "cad", "solidworks"],
    "English":       ["tieng anh", "english", "ielts", "toeic"],
    "Communication": ["giao tiep", "communication"],
    "Negotiation":   ["dam phan", "negotiation"],
    "Project Management": ["quan ly du an", "project management", "agile", "scrum"],
}

CORE_SKILLS_CANON = {"Excel", "Tax", "VAT", "Invoice", "Financial Reporting", "Audit", "SAP", "MISA", "AR/AP", "Bookkeeping"}

# LOCATION
VN_CITY_ALIASES = {
    "tp hcm": "Ho Chi Minh City",
    "tphcm": "Ho Chi Minh City",
    "hcm": "Ho Chi Minh City",
    "sai gon": "Ho Chi Minh City",
    "ho chi minh": "Ho Chi Minh City",
    "ha noi": "Ha Noi",
    "hanoi": "Ha Noi",
    "da nang": "Da Nang",
    "danang": "Da Nang",
    "can tho": "Can Tho",
    "hai phong": "Hai Phong",
    "binh duong": "Binh Duong",
    "dong nai": "Dong Nai",
}

DETAIL_CUES = ["quan","q","huyen","phuong","duong","street","ward","district","tp","thi xa","thi tran"]

# ROLE PATTERNS
ROLE_PATTERNS = [
    (r"\bke toan thue\b|\btax accountant\b", "Tax Accountant"),
    (r"\bke toan tong hop\b|\bgeneral accountant\b", "General Accountant"),
    (r"\bke toan noi bo\b|\binternal accountant\b", "Internal Accountant"),
    (r"\bke toan\b|\baccountant\b", "Accountant"),
    (r"\bkiem toan\b|\baudit\b", "Auditor"),
    (r"\bfinance\b|\bfinancial\b|\btai chinh\b|\bngan hang\b", "Finance"),
    (r"\bbusiness analyst\b|\bba\b", "Business Analyst"),
    (r"\bdata\b|\bmachine learning\b|\bai\b", "Data"),
    (r"\bbackend\b|\bfrontend\b|\bfullstack\b|\bdeveloper\b|\bengineer\b|\bsoftware\b|\blap trinh\b", "Software Engineer"),
    (r"\bdesign\b|\bthiet ke\b|\bgraphic\b|\bui/ux\b", "Designer"),
    (r"\bmarketing\b|\bseo\b|\bcontent\b|\btruyen thong\b", "Marketing"),
    (r"\bhr\b|\bnhan su\b|\btuyen dung\b", "HR"),
    (r"\bsales\b|\bkinh doanh\b|\bban hang\b", "Sales"),
    (r"\bphap che\b|\blegal\b|\bluat\b", "Legal"),
    (r"\bmua hang\b|\bprocurement\b|\blogistics\b|\bchuoi cung ung\b", "Supply Chain"),
    (r"\bcham soc khach hang\b|\bcustomer service\b|\btu van\b", "Customer Service"),
    (r"\bbao tri\b|\bky thuat\b|\bco khi\b|\bdien\b|\bvan hanh\b", "Engineering"),
]

# ROLE SIMILARITY
ROLE_SIM = {
    ("Tax Accountant", "General Accountant"): 0.70,
    ("Internal Accountant", "General Accountant"): 0.60,
    ("Accountant", "General Accountant"): 0.55,
    ("Accountant", "Tax Accountant"): 0.45,
    ("Accountant", "Finance"): 0.45,
    ("Data", "Software Engineer"): 0.45,
    ("Business Analyst", "Data"): 0.50,
    ("Business Analyst", "Software Engineer"): 0.40,
    ("Marketing", "Sales"): 0.40,
    ("Customer Service", "Sales"): 0.45,
}

# EXPERIENCE SIMILARITY
EXP_NEAR = {
    ("Exp_0_1","Exp_1_3"): 0.6,
    ("Exp_1_3","Exp_3_5"): 0.6,
    ("Exp_3_5","Exp_5_plus"): 0.5,
}

# SECTION HINTS
SECTION_HINTS = {
    "skills":     ["ky nang", "kỹ năng", "skills", "technical skills", "skill set", "core skills"],
    "experience": ["kinh nghiem", "kinh nghiệm", "experience", "work history", "employment"],
    "projects":   ["du an", "dự án", "projects", "project"],
    "education":  ["hoc van", "học vấn", "education"],
    "summary":    ["tom tat", "tóm tắt", "summary", "profile", "about"],
    "certs":      ["chung chi", "chứng chỉ", "certifications", "certificate"],
}

SECTION_WEIGHT = {
    "skills": 0.70,
    "experience": 0.45,
    "projects": 0.35,
    "certs": 0.40,
    "summary": 0.30,
    "education": 0.15,
    "unknown": 0.25,
}

# JOB INFO COLUMN MAPPING
COL = {
    "job_id": "JobID",
    "job_url": "URL_Job",
    "job_title": "Title",
    "company": "Name company",
    "location": "Job Address",
    "location_detail": "Job Address detail",
    "requirements": "Job Requirements",
    "salary": "Salary",
    "experience": "Experience",
    "job_desc": "Job description",
    "job_type": "Job type",
    "benefit": "benefit"
}

# RDF CLASSES
RDF_CLASSES = [
    "User", "JobPosting", "JobRoleCanonical", "JobRoleRaw", "Company",
    "Location", "ExperienceBucket", "SalaryBucket", "Skill", "SkillRaw",
    "Industry", "Qualification"
]

# RDF OBJECT PROPERTIES
RDF_OBJ_PROPS = [
    "HAS_ROLE_CANONICAL", "HAS_ROLE_RAW", "POSTED_BY", "LOCATED_IN",
    "REQUIRES_EXP_BUCKET", "HAS_SALARY_BUCKET", "REQUIRES_SKILL",
    "HAS_SKILL", "MATCHES_JOB", "SIMILAR_TO", "NORMALIZES_TO",
    "MENTIONS_SKILL_RAW", "BELONGS_TO_INDUSTRY", "REQUIRES_QUALIFICATION",
    "CO_OCCURS_WITH"
]

# LLM-ENHANCED HR KG + GNN CONFIGURATION
DATASET_PATH = "data/jobs_dataset.csv"
GNN_CHECKPOINT_PATH = "data/hr_kg_gnn.pt"
LLM_CACHE_PATH = "data/llm_cache.json"
MAX_JOBS_SERVING = 15000  # Subset cap for interactive Flask serving; train_gnn.py can train on more
GNN_TEXT_DIM = 128
GNN_HIDDEN_DIM = 128
GNN_EPOCHS = 12
GNN_LR = 2e-3
MIN_COOCCUR_SUPPORT = 10

# EVALUATION SETTINGS
SAMPLE_N = 100
TRIALS = 5
KS = [1, 3, 5, 10]

# Add or update configuration variables for graph visualization
RANDOM_SEED = 42  # Seed for reproducibility in layouts
SHOW_EDGE_SCORES = True  # Whether to display edge scores
TOPK_USER_JOB = 3  # Top K jobs to display for a user
TOPK_SIMILAR = 3  # Top K similar nodes to display
CENTER_JOB_ID = None  # Specific job ID to center the graph on
CENTER_MODE = "user"  # Options: "user", "top_job", "job_id", "random_job"
USER_ID = "user::cv_001"  # Default user ID for the graph

# Define important edges for visualization
IMPORTANT_EDGES = ["MATCHES_JOB", "SIMILAR_TO"]
