import re
import pdfplumber


# ================= NUMERIC FEATURE PATTERNS =================

PATTERNS = {

    # ---------- HEART FEATURES ----------
    "age": r"age[:=\s]+(\d+)",
    "trestbps": r"(resting bp|resting blood pressure|bp)[:=\s]+(\d+)",
    "chol": r"(cholesterol|chol)[:=\s]+(\d+)",
    "oldpeak": r"(oldpeak|st depression)[:=\s]+([\d\.]+)",
    "thalach": r"(max heart rate|thalach)[:=\s]+(\d+)",
    "ca": r"(ca|vessels colored)[:=\s]+(\d+)",

    # ---------- CKD FEATURES ----------
    "SerumCreatinine": r"(creatinine|serum creatinine|sc)[:=\s]+([\d\.]+)",
    "BUNLevels": r"(bun|blood urea nitrogen|urea)[:=\s]+([\d\.]+)",
    "GFR": r"(gfr|glomerular filtration rate)[:=\s]+([\d\.]+)",
    "HemoglobinLevels": r"(hemoglobin|hb|hemo)[:=\s]+([\d\.]+)"
}


# ================= KEYWORD FEATURES =================

KEYWORDS = {

    # heart
    "chest pain": ("cp", 2),
    "typical angina": ("cp", 3),
    "angina": ("exang", 1),
    "exercise angina": ("exang", 1),

    # blood pressure hints
    "high bp": ("trestbps", 140),
    "hypertension": ("trestbps", 150),

    # CKD related
    "anemia": ("HemoglobinLevels", 10.5),
    "kidney disease": ("GFR", 40)
}


# ================= EXTRACT TEXT =================

def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text.lower() + " "

    return text


# ================= NUMERIC EXTRACTION =================

def extract_numeric_features(text):

    features = {}

    for key, pattern in PATTERNS.items():

        match = re.search(pattern, text)

        if match:

            # last group contains the number
            value = match.groups()[-1]

            try:
                features[key] = float(value)
            except:
                pass

    return features


# ================= KEYWORD EXTRACTION =================

def keywords_to_features(text):

    features = {}

    for keyword, (feature, value) in KEYWORDS.items():

        if keyword in text:
            features[feature] = value

    return features


# ================= MAIN PARSER =================

def parse_medical_report(file):

    text = extract_text_from_pdf(file)

    numeric = extract_numeric_features(text)

    keywords = keywords_to_features(text)

    features = {**keywords, **numeric}

    return features