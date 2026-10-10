import os
import re
import json
import time
import uuid
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse, urljoin, quote_plus
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from groq import Groq

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

try:
    from googlesearch import search as google_search
    GOOGLE_OK = True
except ImportError:
    GOOGLE_OK = False

try:
    from ddgs import DDGS
    DDGS_OK = True
except ImportError:
    DDGS_OK = False

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Check your .env file or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ============================================================
# SEARCH CACHE + RATE LIMITER
# ============================================================
SEARCH_CACHE = {}
GOOGLE_BLOCKED_UNTIL = 0
LAST_CALL_TIME = 0


def _throttle():
    global LAST_CALL_TIME
    elapsed = time.time() - LAST_CALL_TIME
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    LAST_CALL_TIME = time.time()


# ============================================================
# HISTORY
# ============================================================
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)


def save_history(industry, location, keywords, df):
    try:
        sid = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        data = {"id": sid, "timestamp": ts, "industry": industry,
                "location": location, "keywords": keywords,
                "lead_count": len(df), "leads": df.to_dict(orient="records")}
        with open(os.path.join(HISTORY_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return sid
    except Exception as e:
        st.warning(f"Could not save history: {e}")
        return None


def load_all_history():
    items = []
    try:
        for fname in os.listdir(HISTORY_DIR):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
                    items.append(json.load(f))
            except Exception: continue
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except Exception: pass
    return items


def delete_history(sid):
    try:
        p = os.path.join(HISTORY_DIR, f"{sid}.json")
        if os.path.exists(p): os.remove(p)
        return True
    except Exception: return False


def clear_all_history():
    try:
        for f in os.listdir(HISTORY_DIR):
            if f.endswith(".json"): os.remove(os.path.join(HISTORY_DIR, f))
        return True
    except Exception: return False


def build_previous_leads_index():
    seen = {}
    try:
        for fname in os.listdir(HISTORY_DIR):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception: continue
            ts = data.get("timestamp", "")
            context = f"{data.get('industry','')} · {data.get('location','')}"
            for lead in data.get("leads", []):
                n = str(lead.get("Business Name","")).strip().lower()
                w = str(lead.get("Website","")).strip().lower()
                for k in ([f"name:{n}"] if n else []) + ([f"site:{w}"] if w else []):
                    if k not in seen: seen[k] = {"timestamp": ts, "context": context}
    except Exception: pass
    return seen


def mark_duplicates(df, prev_index):
    dups, first = [], []
    for _, row in df.iterrows():
        n = str(row.get("Business Name","")).strip().lower()
        w = str(row.get("Website","")).strip().lower()
        hit = None
        if n and f"name:{n}" in prev_index: hit = prev_index[f"name:{n}"]
        elif w and f"site:{w}" in prev_index: hit = prev_index[f"site:{w}"]
        if hit:
            dups.append("Yes"); first.append(f"{hit['timestamp']} · {hit['context']}")
        else:
            dups.append("No"); first.append("")
    df = df.copy()
    df["Duplicate"] = dups
    df["First Seen"] = first
    return df


# ============================================================
# COLUMNS
# ============================================================
COLUMNS = [
    "Business Name", "Business Category", "Lead Score", "AI Summary",
    "Phone Number", "Email", "Website",
    "Street Address", "City", "State", "ZIP Code", "Country",
    "Google Maps URL", "Rating", "Review Count",
    "Facebook Page", "Instagram", "LinkedIn Company URL", "YouTube", "Twitter",
    "Executive Name", "Executive Title", "Executive LinkedIn",
    "Executive Email", "Email Pattern", "Executive Source",
    "Source URL",
    "Audit Score", "Audit Grade", "Issues", "Pitch Angle",
    "Duplicate", "First Seen"
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]

EMAIL_TEMPLATES = {
    "Cold Intro": "Short personalized cold outreach. Mention industry + something from AI summary. Under 120 words. Soft CTA for 15-min call.",
    "Follow-up": "Polite follow-up. Reference previous message, restate value briefly, gently ask if interested. Under 100 words.",
    "LinkedIn Message": "Short LinkedIn connection message (<300 chars). Friendly, mention industry, end with soft question.",
    "Partnership Pitch": "Partnership pitch. Explain mutual benefits, reference business + industry, suggest short call. Under 150 words.",
    "Re-engagement": "Re-engage a past contact who went quiet. Warm but not pushy. Mention something new. Under 100 words.",
}

TONES = ["Professional", "Friendly", "Casual", "Direct"]


# ============================================================
# SEARCH LAYER
# ============================================================
def ddg_web_search(query, max_results=10):
    results = []
    if not DDGS_OK: return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title",""),
                                "href": r.get("href",""),
                                "body": r.get("body","")})
    except Exception: pass
    return results


def google_web_search(query, max_results=10):
    global GOOGLE_BLOCKED_UNTIL
    results = []
    if not GOOGLE_OK: return results
    if time.time() < GOOGLE_BLOCKED_UNTIL:
        return results
    try:
        for url in google_search(query, num_results=max_results, lang="en",
                                  region="us", sleep_interval=3, advanced=False):
            results.append({"title": "", "href": url, "body": ""})
            if len(results) >= max_results: break
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg or "sorry/index" in msg:
            GOOGLE_BLOCKED_UNTIL = time.time() + 600
    return results


def bing_web_search(query, max_results=10):
    results = []
    try:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200: return results
        soup = BeautifulSoup(r.text, "lxml")
        for li in soup.select("li.b_algo")[:max_results]:
            a = li.find("a", href=True)
            if not a: continue
            title = a.get_text(strip=True)
            href = a["href"]
            snippet_tag = li.find("p")
            body = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if href.startswith("http"):
                results.append({"title": title, "href": href, "body": body})
    except Exception: pass
    return results


def web_search(query, max_results=10):
    qkey = query.strip().lower()
    if qkey in SEARCH_CACHE:
        return SEARCH_CACHE[qkey][:max_results]
    results = []
    _throttle()
    results = ddg_web_search(query, max_results)
    if results:
        SEARCH_CACHE[qkey] = results
        return results
    _throttle()
    results = bing_web_search(query, max_results)
    if results:
        SEARCH_CACHE[qkey] = results
        return results
    _throttle()
    results = google_web_search(query, max_results)
    SEARCH_CACHE[qkey] = results
    return results


def multi_search(industry, location, target):
    queries = [
        f"{industry} in {location}",
        f"best {industry} {location}",
        f"top {industry} companies {location} contact",
        f"{industry} {location} phone number address",
        f"{industry} near {location} website",
    ]
    all_r, seen = [], set()
    for q in queries:
        if len(all_r) >= target * 3: break
        for r in web_search(q, max_results=10):
            if r.get("href") and r["href"] not in seen:
                seen.add(r["href"]); all_r.append(r)
        time.sleep(0.5)
    return all_r


PROMPT = """You are a lead data extractor. Return ONLY a JSON object.

Industry: {industry}
Location: {location}
Keywords (optional): {keywords}

RULES:
1. Extract EVERY distinct business.
2. Never invent data. Use "" if a field is missing.
3. Return UP TO {limit} businesses.
4. "Lead Score" = INTEGER 1-10 as string.
5. "AI Summary" = 1 short sentence (max 20 words).

Return JSON with key "leads" → array of objects with these keys:
"Business Name", "Business Category", "Lead Score", "AI Summary",
"Phone Number", "Email", "Website",
"Street Address", "City", "State", "ZIP Code", "Country",
"Google Maps URL", "Rating", "Review Count",
"Facebook Page", "Instagram", "LinkedIn Company URL", "YouTube", "Twitter",
"Source URL"

Search results:
{results}
"""


def call_groq(prompt, json_mode=False, temperature=0.1):
    last_err = None
    for model in GROQ_MODELS:
        try:
            kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        except Exception as e:
            last_err = f"{model}: {e}"; continue
    raise RuntimeError(f"All Groq models failed. Last: {last_err}")


def extract_leads(industry, location, keywords, results, limit):
    if not results: return []
    trimmed = results[:20]
    rtext = "\n\n".join(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                        for r in trimmed)
    prompt = PROMPT.format(industry=industry, location=location,
                           keywords=keywords or "None", limit=limit, results=rtext)
    for jm in (True, False):
        try:
            text = call_groq(prompt, json_mode=jm)
            text = re.sub(r"^```(json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            parsed = json.loads(text)
            data = (parsed.get("leads") or parsed.get("Leads") or []) if isinstance(parsed, dict) else parsed
            if data: return data
        except Exception as e:
            st.warning(f"Extract attempt failed: {e}"); continue
    st.error("Could not extract any leads.")
    return []


# ============================================================
# HELPERS — URL CLEANING + COUNTRY DETECTION
# ============================================================
SOCIAL_PATTERNS = {
    "Facebook":  r"(?:facebook\.com|fb\.com)/([A-Za-z0-9._\-/]+)",
    "Instagram": r"instagram\.com/([A-Za-z0-9._\-/]+)",
    "LinkedIn":  r"linkedin\.com/(?:company|in)/([A-Za-z0-9._\-/]+)",
    "YouTube":   r"youtube\.com/(?:c/|channel/|user/|@)?([A-Za-z0-9._\-/]+)",
    "Twitter":   r"(?:twitter\.com|x\.com)/([A-Za-z0-9._\-/]+)",
}
PHONE_REGEX = re.compile(r"(?:\+?1[\s\-.]?)?\(?\b([2-9]\d{2})\)?[\s\-.]?(\d{3})[\s\-.]?(\d{4})\b")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

JUNK_EMAILS = ["example", "sentry", "yourdomain", "@2x", ".png", ".jpg", ".jpeg",
               ".gif", ".webp", "wixpress", "sentry.io", "w3.org"]

TITLE_KEYWORDS = [
    "owner", "co-owner", "founder", "co-founder", "ceo", "chief executive",
    "president", "managing director", "director", "manager", "general manager",
    "partner", "principal", "proprietor", "head of", "vp", "vice president",
    "cmo", "cfo", "coo", "cto", "medical director", "practice manager",
]

# ---- Facebook URL cleaning ----
FB_JUNK_SEGMENTS = {
    "sharer", "sharer.php", "plugins", "login", "login.php", "dialog",
    "tr", "profile.php", "pages", "events", "groups", "photo.php",
    "story.php", "permalink.php", "watch", "marketplace", "gaming",
    "help", "policies", "about", "legal", "careers", "business", "developers"
}


def clean_facebook_url(url):
    """
    Return a canonical Facebook business page URL or ''.
    Strips share links, plugins, login redirects, and query params.
    """
    if not url: return ""
    try:
        # Strip query params first
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        if not path: return ""

        first_seg = path.split("/")[0].lower()
        if first_seg in FB_JUNK_SEGMENTS:
            return ""

        # Keep only /pagename or /pagename/subsection
        parts = [seg for seg in path.split("/") if seg and seg.lower() not in FB_JUNK_SEGMENTS]
        if not parts: return ""

        # Facebook page handles are typically 3-60 chars, no dots or spaces
        handle = parts[0]
        if len(handle) < 3 or len(handle) > 60: return ""
        if "." in handle or " " in handle: return ""
        # Common junk
        if handle.lower() in ("home", "welcome", "index", "en", "us"): return ""

        return f"https://www.facebook.com/{handle}"
    except Exception:
        return ""


def clean_linkedin_company_url(url):
    """Return a canonical LinkedIn company URL or '' (rejects profile URLs)."""
    if not url: return ""
    try:
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        # Prefer /company/xyz
        m = re.match(r"company/([A-Za-z0-9\-_%]+)", path, re.IGNORECASE)
        if m:
            handle = m.group(1)
            if len(handle) > 1:
                return f"https://www.linkedin.com/company/{handle}"
        return ""
    except Exception:
        return ""


def clean_instagram_url(url):
    if not url: return ""
    try:
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        if not path: return ""
        handle = path.split("/")[0]
        if handle.lower() in ("p", "explore", "reel", "reels", "stories", "accounts"):
            return ""
        if len(handle) < 2 or len(handle) > 40: return ""
        return f"https://www.instagram.com/{handle}"
    except Exception:
        return ""


# ---- Country detection ----
COUNTRY_HINTS = {
    "united states": ["usa", "united states", "u.s.a", "us"],
    "united kingdom": ["uk", "united kingdom", "england", "scotland", "wales", "great britain"],
    "canada": ["canada", "ca"],
    "australia": ["australia", "au"],
    "india": ["india", "in"],
    "pakistan": ["pakistan", "pk"],
    "germany": ["germany", "de", "deutschland"],
    "france": ["france", "fr"],
    "spain": ["spain", "es", "españa"],
    "italy": ["italy", "it", "italia"],
    "netherlands": ["netherlands", "nl", "holland"],
    "uae": ["uae", "united arab emirates", "dubai", "abu dhabi"],
    "saudi arabia": ["saudi arabia", "ksa", "saudi"],
    "singapore": ["singapore", "sg"],
    "japan": ["japan", "jp"],
    "china": ["china", "cn"],
    "brazil": ["brazil", "br", "brasil"],
    "mexico": ["mexico", "mx", "méxico"],
    "new zealand": ["new zealand", "nz"],
    "ireland": ["ireland", "ie"],
    "south africa": ["south africa", "za"],
}

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}


def detect_country(lead):
    """
    Detect country using multiple signals:
    1. Explicit Country field
    2. State abbreviation (US)
    3. ZIP code format (US: 5 digits)
    4. Phone prefix
    5. Domain TLD
    6. AI summary text
    """
    try:
        # 1. Explicit
        c = str(lead.get("Country", "")).strip()
        if c:
            for known, aliases in COUNTRY_HINTS.items():
                if c.lower() in aliases or c.lower() == known:
                    return known.title()

        # 2. State = US abbreviation
        state = str(lead.get("State", "")).strip().upper()
        if state in US_STATES:
            return "United States"

        # 3. ZIP = 5 digits → US
        zipc = str(lead.get("ZIP Code", "")).strip()
        if re.match(r"^\d{5}(-\d{4})?$", zipc):
            return "United States"

        # 4. Phone prefix
        phone = str(lead.get("Phone Number", "")).strip()
        if phone and phone != "N/A":
            d = re.sub(r"\D", "", phone)
            if d.startswith("1") and len(d) == 11:
                return "United States"
            if d.startswith("44"): return "United Kingdom"
            if d.startswith("91"): return "India"
            if d.startswith("92"): return "Pakistan"
            if d.startswith("61"): return "Australia"
            if d.startswith("49"): return "Germany"
            if d.startswith("33"): return "France"
            if d.startswith("86"): return "China"
            if d.startswith("81"): return "Japan"
            if d.startswith("65"): return "Singapore"
            if d.startswith("971"): return "UAE"

        # 5. Domain TLD
        website = str(lead.get("Website", "")).strip()
        if website:
            tld = urlparse(website).netloc.rsplit(".", 1)[-1].lower()
            tld_map = {
                "uk": "United Kingdom", "ca": "Canada", "au": "Australia",
                "in": "India", "pk": "Pakistan", "de": "Germany",
                "fr": "France", "es": "Spain", "it": "Italy",
                "nl": "Netherlands", "ae": "UAE", "sa": "Saudi Arabia",
                "sg": "Singapore", "jp": "Japan", "cn": "China",
                "br": "Brazil", "mx": "Mexico", "nz": "New Zealand",
                "ie": "Ireland", "za": "South Africa",
            }
            if tld in tld_map:
                return tld_map[tld]

        # 6. AI summary text
        summary = (str(lead.get("AI Summary", "")) + " " + str(lead.get("Street Address", ""))).lower()
        for known, aliases in COUNTRY_HINTS.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", summary):
                    return known.title()

    except Exception:
        pass
    return ""


def safe_get(url, timeout=8):
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except Exception:
        return None


def domain_from_url(url):
    try:
        p = urlparse(url)
        d = p.netloc.lower()
        if d.startswith("www."): d = d[4:]
        return d
    except Exception:
        return ""


def extract_jsonld(soup):
    data = {}
    try:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(script.string or "")
            except Exception:
                continue
            items = obj if isinstance(obj, list) else [obj]
            for item in items:
                if not isinstance(item, dict): continue
                t = item.get("@type", "")
                if t in ("LocalBusiness", "Organization", "Store", "MedicalBusiness",
                         "HealthAndBeautyBusiness", "ProfessionalService", "Restaurant"):
                    addr = item.get("address", {})
                    if isinstance(addr, dict):
                        data.setdefault("Street Address", addr.get("streetAddress", ""))
                        data.setdefault("City", addr.get("addressLocality", ""))
                        data.setdefault("State", addr.get("addressRegion", ""))
                        data.setdefault("ZIP Code", addr.get("postalCode", ""))
                        data.setdefault("Country", addr.get("addressCountry", ""))
                    elif isinstance(addr, str):
                        data.setdefault("Street Address", addr)
                    data.setdefault("Phone Number", item.get("telephone", ""))
                    data.setdefault("Email", item.get("email", ""))
                    geo = item.get("geo", {})
                    if isinstance(geo, dict):
                        lat, lng = geo.get("latitude"), geo.get("longitude")
                        if lat and lng:
                            data.setdefault("Google Maps URL",
                                            f"https://www.google.com/maps?q={lat},{lng}")
                    rating = item.get("aggregateRating", {})
                    if isinstance(rating, dict):
                        data.setdefault("Rating", str(rating.get("ratingValue", "")))
                        data.setdefault("Review Count", str(rating.get("reviewCount", "")))
                    for same in item.get("sameAs", []) or []:
                        low = same.lower()
                        if "facebook.com" in low:
                            cl = clean_facebook_url(same)
                            if cl: data.setdefault("Facebook Page", cl)
                        elif "instagram.com" in low:
                            cl = clean_instagram_url(same)
                            if cl: data.setdefault("Instagram", cl)
                        elif "linkedin.com/company" in low:
                            cl = clean_linkedin_company_url(same)
                            if cl: data.setdefault("LinkedIn Company URL", cl)
                        elif "youtube.com" in low:
                            data.setdefault("YouTube", same)
                        elif "twitter.com" in low or "x.com" in low:
                            data.setdefault("Twitter", same)
                    if item.get("founder"):
                        f = item["founder"]
                        if isinstance(f, dict):
                            data.setdefault("_exec_name", f.get("name", ""))
                        elif isinstance(f, str):
                            data.setdefault("_exec_name", f)
    except Exception: pass
    return data


def extract_contacts_from_html(html, base_url):
    result = {}
    soup = BeautifulSoup(html, "lxml")

    # tel: and mailto: links
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith("tel:"):
            d = re.sub(r"\D", "", h[4:])
            if len(d) >= 10:
                result.setdefault("Phone Number", h[4:].strip())
        elif h.startswith("mailto:"):
            e = h[7:].split("?")[0].strip()
            if EMAIL_REGEX.match(e) and not any(x in e.lower() for x in JUNK_EMAILS):
                result.setdefault("Email", e)

    # Social links with cleanup
    for a in soup.find_all("a", href=True):
        h = a["href"]
        low = h.lower()
        if "facebook.com" in low or "fb.com" in low:
            cl = clean_facebook_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("Facebook Page", cl)
        elif "instagram.com" in low:
            cl = clean_instagram_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("Instagram", cl)
        elif "linkedin.com/company" in low:
            cl = clean_linkedin_company_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("LinkedIn Company URL", cl)
        elif "youtube.com" in low:
            full = h if h.startswith("http") else urljoin(base_url, h)
            result.setdefault("YouTube", full)
        elif "twitter.com" in low or "x.com" in low:
            full = h if h.startswith("http") else urljoin(base_url, h)
            result.setdefault("Twitter", full)

    # <address> tag
    addr = soup.find("address")
    if addr:
        t = addr.get_text(separator=" ", strip=True)
        if len(t) > 10: result.setdefault("_address_block", t)

    # JSON-LD
    jsonld = extract_jsonld(soup)
    for k, v in jsonld.items():
        if v and k not in result: result[k] = v

    # Text fallbacks
    page_text = soup.get_text(separator=" ", strip=True)
    if "Phone Number" not in result:
        phones = PHONE_REGEX.findall(page_text)
        if phones:
            d = phones[0]; result["Phone Number"] = f"({d[0]}) {d[1]}-{d[2]}"
    if "Email" not in result:
        emails = [e for e in EMAIL_REGEX.findall(page_text)
                  if not any(x in e.lower() for x in JUNK_EMAILS)]
        if emails: result["Email"] = emails[0]

    # Google Maps
    for a in soup.find_all("a", href=True):
        if "google.com/maps" in a["href"] or "maps.google" in a["href"]:
            result.setdefault("Google Maps URL", a["href"]); break

    return result


# ============================================================
# EXECUTIVE FINDER
# ============================================================
def find_executives_from_html(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    executives = []

    for kw in TITLE_KEYWORDS:
        pattern = re.compile(
            rf"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,2}})\s*[,\-–|]\s*([A-Za-z\s]{{2,40}}?{kw}[A-Za-z\s]{{0,20}})",
            re.IGNORECASE
        )
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            title = match.group(2).strip()
            if len(name.split()) < 2 or len(name) > 40: continue
            if any(x in name.lower() for x in ["privacy", "policy", "terms", "cookie"]): continue
            executives.append({"name": name, "title": title.title(), "source": base_url})
            if len(executives) >= 5: break
        if len(executives) >= 5: break

    if not executives:
        for kw in ["founder", "ceo", "owner", "president"]:
            p = re.compile(rf"{kw}[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)", re.IGNORECASE)
            for m in p.finditer(text):
                executives.append({"name": m.group(1), "title": kw.title(), "source": base_url})
                if len(executives) >= 3: break
            if executives: break
    return executives


def google_search_executive(business_name, city="", state=""):
    if not business_name: return {}
    info = {}
    q1 = f'"{business_name}" (CEO OR Founder OR Owner OR President OR Manager)'
    if city: q1 += f" {city}"
    results = web_search(q1, max_results=5)
    combined = " ".join([f"{r.get('title','')} {r.get('body','')}" for r in results])

    for kw in ["CEO", "Founder", "Co-Founder", "Owner", "President",
               "Managing Director", "Director", "Manager", "Partner"]:
        pat = re.compile(rf"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,\-–]\s*{kw}", re.IGNORECASE)
        m = pat.search(combined)
        if m:
            info["Executive Name"] = m.group(1).strip()
            info["Executive Title"] = kw.title()
            break
    if "Executive Name" not in info:
        for kw in ["CEO", "Founder", "Owner", "President"]:
            pat = re.compile(rf"{kw}\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", re.IGNORECASE)
            m = pat.search(combined)
            if m:
                info["Executive Name"] = m.group(1).strip()
                info["Executive Title"] = kw.title()
                break

    q2 = f'"{business_name}" site:linkedin.com/in'
    if city: q2 += f" {city}"
    li_results = web_search(q2, max_results=3)
    for r in li_results:
        href = r.get("href", "")
        if "linkedin.com/in/" in href:
            info["Executive LinkedIn"] = href
            title = r.get("title", "")
            m = re.match(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", title)
            if m and "Executive Name" not in info:
                info["Executive Name"] = m.group(1)
            break
    return info


def find_company_socials(business_name, city="", state=""):
    """Search for the official Facebook + LinkedIn company URLs."""
    result = {}
    if not business_name: return result

    # Facebook
    q_fb = f'"{business_name}" facebook'
    if city: q_fb += f" {city}"
    for r in web_search(q_fb, max_results=4):
        url = r.get("href", "")
        if "facebook.com" in url:
            cl = clean_facebook_url(url)
            if cl:
                result["Facebook Page"] = cl
                break

    # LinkedIn company
    q_li = f'"{business_name}" site:linkedin.com/company'
    if city: q_li += f" {city}"
    for r in web_search(q_li, max_results=4):
        url = r.get("href", "")
        if "linkedin.com/company" in url:
            cl = clean_linkedin_company_url(url)
            if cl:
                result["LinkedIn Company URL"] = cl
                break

    return result


def detect_email_pattern(domain, business_name):
    if not domain: return "", ""
    q = f'"@{domain}"'
    results = web_search(q, max_results=5)
    combined = " ".join([f"{r.get('title','')} {r.get('body','')}" for r in results])
    emails = [e for e in EMAIL_REGEX.findall(combined)
              if domain in e.lower() and not any(x in e.lower() for x in JUNK_EMAILS)]
    pattern = ""
    if emails:
        sample = emails[0]
        local = sample.split("@")[0].lower()
        if "." in local and local.count(".") == 1:
            pattern = "{first}.{last}"
        elif local.isalpha() and len(local) > 3:
            pattern = "{first}"
        elif len(local) == 1:
            pattern = "{f}{last}"
        else:
            pattern = "{first}"
    else:
        pattern = "{first}"
    return pattern, emails[0] if emails else ""


def infer_email(pattern, name, domain):
    if not name or not domain or not pattern: return ""
    parts = name.strip().split()
    if len(parts) < 2: return ""
    first = re.sub(r"[^a-z]", "", parts[0].lower())
    last = re.sub(r"[^a-z]", "", parts[-1].lower())
    f = first[0] if first else ""
    try:
        local = pattern.format(first=first, last=last, f=f)
    except Exception:
        local = first
    return f"{local}@{domain}"


# ============================================================
# WEBSITE RESEARCH
# ============================================================
CONTACT_PATHS = ["", "/contact", "/contact-us", "/about", "/about-us"]
TEAM_PATHS = ["/team", "/our-team", "/staff", "/leadership", "/management",
              "/about/team", "/owners", "/meet-the-team"]


def research_website(url):
    if not url or not url.startswith("http"): return {}
    collected, visited = {}, set()
    base = url.rstrip("/")
    for path in CONTACT_PATHS:
        target = base + path if path else base
        if target in visited: continue
        visited.add(target)
        resp = safe_get(target, timeout=8)
        if not resp or resp.status_code != 200: continue
        info = extract_contacts_from_html(resp.text, target)
        for k, v in info.items():
            if v and k not in collected: collected[k] = v
        important = ["Phone Number", "Email", "Street Address", "Facebook Page",
                     "Instagram", "LinkedIn Company URL"]
        if all(k in collected for k in important): break
        time.sleep(0.3)
    return collected


def find_executives(url):
    if not url or not url.startswith("http"): return []
    base = url.rstrip("/")
    all_execs = []
    for path in TEAM_PATHS + ["/about", "/about-us"]:
        target = base + path
        resp = safe_get(target, timeout=8)
        if not resp or resp.status_code != 200: continue
        found = find_executives_from_html(resp.text, target)
        for e in found:
            if not any(x["name"].lower() == e["name"].lower() for x in all_execs):
                all_execs.append(e)
        if len(all_execs) >= 3: break
        time.sleep(0.3)
    return all_execs


def google_search_fallback(business_name, city, state):
    result = {}
    if not business_name: return result
    q = f'"{business_name}" {city} {state} (phone OR contact OR email OR address)'.strip()
    results = web_search(q, max_results=5)
    combined = " ".join([f"{r.get('title','')} {r.get('body','')}" for r in results])
    phones = PHONE_REGEX.findall(combined)
    if phones:
        d = phones[0]; result["Phone Number"] = f"({d[0]}) {d[1]}-{d[2]}"
    emails = [e for e in EMAIL_REGEX.findall(combined)
              if not any(x in e.lower() for x in JUNK_EMAILS)]
    if emails: result["Email"] = emails[0]
    for r in results:
        if "google.com/maps" in r.get("href", ""):
            result["Google Maps URL"] = r["href"]; break
    if "Google Maps URL" not in result and city:
        q2 = f"{business_name} {city} {state}".replace(" ", "+")
        result["Google Maps URL"] = f"https://www.google.com/maps/search/{q2}"
    return result


ENRICH_PROMPT = """Extract contact details from this business website's text.

Business: {name}
Website: {website}

Text:
-----
{page_text}
-----

Return ONLY JSON with these keys (use "" if missing):
"Phone Number", "Email", "Street Address", "City", "State", "ZIP Code", "Country",
"Facebook Page", "Instagram", "LinkedIn Company URL", "YouTube", "Twitter", "Google Maps URL"
"""


def fetch_website_text_multi(url, max_chars=4000):
    if not url or not url.startswith("http"): return ""
    combined = ""
    for path in ["", "/contact", "/contact-us", "/about", "/about-us"]:
        target = url.rstrip("/") + path
        resp = safe_get(target, timeout=8)
        if resp and resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "noscript"]): tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            combined += f"\n\n--- {target} ---\n{text}"
            if len(combined) > max_chars * 2: break
        time.sleep(0.2)
    return combined[:max_chars * 2]


def enrich_lead_deep(lead):
    website = lead.get("Website", "")
    name = lead.get("Business Name", "")
    city = lead.get("City", "")
    state = lead.get("State", "")
    domain = domain_from_url(website)

    # 1. Crawl website
    site_data = research_website(website)

    # 2. AI parse text
    if website:
        page_text = fetch_website_text_multi(website)
        if page_text:
            try:
                prompt = ENRICH_PROMPT.format(name=name, website=website,
                                              page_text=page_text[:6000])
                text = call_groq(prompt)
                text = re.sub(r"^```(json)?", "", text).strip()
                text = re.sub(r"```$", "", text).strip()
                ai_data = json.loads(text)
                for k, v in ai_data.items():
                    if v and not site_data.get(k): site_data[k] = v
            except Exception: pass

    # 3. Google fallback
    missing = [k for k in ["Phone Number", "Email", "Street Address", "Google Maps URL"]
               if not site_data.get(k)]
    if missing and name:
        fb = google_search_fallback(name, city, state)
        for k, v in fb.items():
            if v and not site_data.get(k): site_data[k] = v

    # 4. Social URL search (Facebook Page + LinkedIn Company URL)
    if not site_data.get("Facebook Page") or not site_data.get("LinkedIn Company URL"):
        socials = find_company_socials(name, city, state)
        for k, v in socials.items():
            if v and not site_data.get(k): site_data[k] = v

    # Clean & normalize any social URLs we have so far
    if site_data.get("Facebook Page"):
        site_data["Facebook Page"] = clean_facebook_url(site_data["Facebook Page"])
    if site_data.get("LinkedIn Company URL"):
        site_data["LinkedIn Company URL"] = clean_linkedin_company_url(site_data["LinkedIn Company URL"])
    if site_data.get("Instagram"):
        site_data["Instagram"] = clean_instagram_url(site_data["Instagram"])

    # 5. Executive discovery
    exec_data = {}
    site_execs = find_executives(website) if website else []
    if site_execs:
        top = site_execs[0]
        exec_data["Executive Name"] = top["name"]
        exec_data["Executive Title"] = top["title"]
        exec_data["Executive Source"] = top["source"]

    google_exec = google_search_executive(name, city, state)
    for k, v in google_exec.items():
        if v and not exec_data.get(k): exec_data[k] = v
    if "Executive Source" not in exec_data and google_exec.get("Executive Name"):
        exec_data["Executive Source"] = "Search engines"

    # 6. Email pattern
    pattern, sample_email = detect_email_pattern(domain, name)
    exec_data["Email Pattern"] = pattern

    exec_email = ""
    if exec_data.get("Executive Name") and domain and pattern:
        exec_email = infer_email(pattern, exec_data["Executive Name"], domain)
    if site_data.get("Email"):
        exec_data["Executive Email"] = site_data["Email"]
    elif exec_email:
        exec_data["Executive Email"] = exec_email + " (inferred)"
    else:
        exec_data["Executive Email"] = "N/A"

    # 7. Merge
    for k, v in site_data.items():
        if k.startswith("_"): continue
        if k in lead and not lead.get(k) and v: lead[k] = v
        elif k not in lead and v: lead[k] = v
    for k, v in exec_data.items():
        if v and not lead.get(k): lead[k] = v

    if "_address_block" in site_data and not lead.get("Street Address"):
        lead["Street Address"] = site_data["_address_block"][:120]

    if not lead.get("Phone Number"):
        lead["Phone Number"] = "N/A"

    # 8. Country detection (uses all signals)
    detected = detect_country(lead)
    if detected:
        lead["Country"] = detected

    return lead


# ============================================================
# WEBSITE AUDIT
# ============================================================
def audit_website(url):
    result = {"score": 0, "grade": "N/A", "issues": "", "signals": {}}
    if not url or not url.startswith("http"):
        result["issues"] = "No website"; return result
    signals = {"ssl": False, "load_time": None, "has_title": False,
               "has_meta_desc": False, "has_viewport": False,
               "has_phone": False, "has_email": False, "modern": 0, "size_kb": 0}
    try:
        signals["ssl"] = urlparse(url).scheme == "https"
        t0 = time.time()
        resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        signals["load_time"] = round(time.time() - t0, 2)
        signals["size_kb"] = round(len(resp.content) / 1024, 1)
        soup = BeautifulSoup(resp.text, "lxml")
        signals["has_title"] = bool(soup.find("title") and soup.find("title").text.strip())
        signals["has_meta_desc"] = bool(soup.find("meta", attrs={"name": "description"}))
        signals["has_viewport"] = bool(soup.find("meta", attrs={"name": "viewport"}))
        t = soup.get_text(separator=" ", strip=True)
        signals["has_phone"] = bool(PHONE_REGEX.search(t))
        signals["has_email"] = bool(EMAIL_REGEX.search(t))
        hl = resp.text.lower()
        m = 0
        if "google fonts" in hl or "fonts.googleapis" in hl: m += 1
        if "bootstrap" in hl or "tailwind" in hl: m += 1
        if "flex" in hl or "grid" in hl: m += 1
        if "@media" in hl: m += 1
        if "<section" in hl or "<main" in hl: m += 1
        signals["modern"] = m
    except requests.exceptions.Timeout:
        result["issues"] = "Site timed out"; return result
    except Exception as e:
        result["issues"] = f"Load error: {str(e)[:60]}"; return result
    score, issues = 0, []
    if signals["ssl"]: score += 15
    else: issues.append("No SSL")
    if signals["load_time"] is not None:
        if signals["load_time"] < 3: score += 20
        elif signals["load_time"] < 6: score += 10; issues.append(f"Slow ({signals['load_time']}s)")
        else: issues.append(f"Very slow ({signals['load_time']}s)")
    if signals["size_kb"] < 1024: score += 10
    else: issues.append(f"Heavy ({signals['size_kb']}KB)")
    if signals["has_title"]: score += 10
    else: issues.append("No title")
    if signals["has_meta_desc"]: score += 10
    else: issues.append("No meta desc")
    if signals["has_viewport"]: score += 15
    else: issues.append("Not mobile")
    if signals["has_phone"]: score += 5
    else: issues.append("No phone on page")
    if signals["has_email"]: score += 5
    else: issues.append("No email on page")
    score += min(signals["modern"] * 2, 10)
    if signals["modern"] < 2: issues.append("Outdated design")
    score = max(0, min(100, score))
    result["score"] = score
    result["signals"] = signals
    if score >= 90: g = "A"
    elif score >= 75: g = "B"
    elif score >= 60: g = "C"
    elif score >= 40: g = "D"
    else: g = "F"
    result["grade"] = g
    result["issues"] = "; ".join(issues) if issues else "No major issues"
    return result


def generate_pitch_angle(lead, audit):
    try:
        prompt = f"""Sales strategist: write ONE short sentence (max 20 words) suggesting the best pitch angle.

Business: {lead.get('Business Name','')}
Industry: {lead.get('Business Category','')}
Audit: {audit.get('score','N/A')}/100
Issues: {audit.get('issues','')}

Return ONLY the sentence."""
        return call_groq(prompt, temperature=0.4).strip()
    except Exception: return ""


def run_full_audit_on_df(df, cb=None):
    total = len(df)
    sc, gr, iss, ang = [], [], [], []
    for i, row in df.iterrows():
        ex = str(row.get("Audit Score","")).strip()
        if ex and ex not in ("", "0"):
            sc.append(row.get("Audit Score")); gr.append(row.get("Audit Grade"))
            iss.append(row.get("Issues","")); ang.append(row.get("Pitch Angle",""))
            if cb: cb(i+1, total, "cached")
            continue
        w = row.get("Website", "")
        if w and w.startswith("http"):
            a = audit_website(w)
            sc.append(a["score"]); gr.append(a["grade"]); iss.append(a["issues"])
            ang.append(generate_pitch_angle(row.to_dict(), a))
        else:
            sc.append(0); gr.append("N/A"); iss.append("No website"); ang.append("")
        if cb: cb(i+1, total, "audited")
    df = df.copy()
    df["Audit Score"] = sc; df["Audit Grade"] = gr
    df["Issues"] = iss; df["Pitch Angle"] = ang
    return df


# ============================================================
# UTILITIES
# ============================================================
def normalize_phone(p):
    if not p or p == "N/A": return p or ""
    d = re.sub(r"\D", "", str(p))
    if len(d) == 10: return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    if len(d) == 11 and d.startswith("1"): return f"({d[1:4]}) {d[4:7]}-{d[7:]}"
    return str(p).strip()


def clean_df(rows):
    for row in rows:
        for c in COLUMNS:
            if c not in row: row[c] = ""
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["Phone Number"] = df["Phone Number"].apply(normalize_phone)
    df = df[(df["Business Name"] != "") | (df["Website"] != "")]
    df = df.drop_duplicates(subset=["Business Name", "Website"], keep="first")
    return df.reset_index(drop=True)


def push_to_google_sheets(df, sheet_name, creds):
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        c = ServiceAccountCredentials.from_json_keyfile_dict(creds, scope)
        gc = gspread.authorize(c)
        try: s = gc.open(sheet_name).sheet1
        except gspread.SpreadsheetNotFound: s = gc.create(sheet_name).sheet1
        s.clear()
        s.update([df.columns.values.tolist()] + df.values.tolist())
        return True, f"Pushed {len(df)} leads to '{sheet_name}'"
    except Exception as e:
        return False, f"Sheets error: {e}"


def ask_ai_about_leads(q, df):
    txt = df.head(30).to_csv(index=False)
    prompt = f"""You are a lead gen assistant.

Data:
---LEADS---
{txt}
---LEADS---

User: "{q}"

Answer concisely using ONLY the data."""
    try: return call_groq(prompt, temperature=0.3)
    except Exception as e: return f"❌ {e}"


def generate_email(tname, tinst, tone, lead):
    s = f"""Business: {lead.get('Business Name','')}
Exec: {lead.get('Executive Name','')} — {lead.get('Executive Title','')}
Category: {lead.get('Business Category','')}
City: {lead.get('City','')}, {lead.get('Country','')}
Audit: {lead.get('Audit Score','')}/100
Issues: {lead.get('Issues','')}
Pitch: {lead.get('Pitch Angle','')}"""
    prompt = f"""Expert sales copywriter.

Write an email using style "{tname}".
Instructions: {tinst}
Tone: {tone}

Lead:
{s}

RULES:
- Address the executive by first name if available.
- Weave in audit issues if relevant.
- Sign "Best regards, [Your Name]".
- Start with "Subject: ..."
- Return ONLY the email."""
    try: return call_groq(prompt, temperature=0.7)
    except Exception as e: return f"❌ {e}"


def generate_pdf_report(df, industry, location, keywords):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    sty = getSampleStyleSheet()
    tstyle = ParagraphStyle("T", parent=sty["Title"], fontSize=24,
                            textColor=colors.HexColor("#FF4B4B"), spaceAfter=20, alignment=1)
    h2 = ParagraphStyle("H2", parent=sty["Heading2"], fontSize=14,
                        textColor=colors.HexColor("#222222"), spaceBefore=12, spaceAfter=6)
    st_ = []
    st_.append(Spacer(1, 1.5*inch))
    st_.append(Paragraph("🎯 Lead Generation Report", tstyle))
    st_.append(Spacer(1, 0.3*inch))
    cover = [["Industry:", industry or "—"], ["Location:", location or "—"],
             ["Keywords:", keywords or "—"],
             ["Generated:", datetime.now().strftime("%B %d, %Y %H:%M")],
             ["Total Leads:", str(len(df))]]
    t = Table(cover, colWidths=[2.2*inch, 3.6*inch])
    t.setStyle(TableStyle([("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                           ("FONTSIZE", (0,0), (-1,-1), 11),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                           ("TOPPADDING", (0,0), (-1,-1), 8)]))
    st_.append(t); st_.append(PageBreak())
    st_.append(Paragraph("📋 Overview", h2))
    rows = [["#", "Business", "Exec", "Title", "Country", "Score"]]
    for i, row in df.iterrows():
        rows.append([str(i+1), str(row.get("Business Name",""))[:28],
                     str(row.get("Executive Name",""))[:20],
                     str(row.get("Executive Title",""))[:16],
                     str(row.get("Country",""))[:14],
                     str(row.get("Lead Score",""))])
    t = Table(rows, colWidths=[0.4*inch, 2.2*inch, 1.5*inch, 1.2*inch, 1.2*inch, 0.6*inch])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FF4B4B")),
                           ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                           ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                           ("FONTSIZE", (0,0), (-1,-1), 8),
                           ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F7F7")]),
                           ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC"))]))
    st_.append(t); st_.append(PageBreak())
    st_.append(Paragraph("🔍 Detailed Profiles", h2))
    for i, row in df.iterrows():
        st_.append(Paragraph(f"#{i+1} · {row.get('Business Name','')}", h2))
        d = [
            ["Category", str(row.get("Business Category",""))],
            ["Lead Score", str(row.get("Lead Score",""))],
            ["AI Summary", str(row.get("AI Summary",""))],
            ["Executive", f"{row.get('Executive Name','')} — {row.get('Executive Title','')}"],
            ["Exec LinkedIn", str(row.get("Executive LinkedIn",""))],
            ["Exec Email", str(row.get("Executive Email",""))],
            ["Email Pattern", str(row.get("Email Pattern",""))],
            ["Phone", str(row.get("Phone Number",""))],
            ["Email", str(row.get("Email",""))],
            ["Website", str(row.get("Website",""))],
            ["Address", f"{row.get('Street Address','')} {row.get('City','')} {row.get('State','')} {row.get('ZIP Code','')}".strip()],
            ["Country", str(row.get("Country",""))],
            ["Google Maps", str(row.get("Google Maps URL",""))],
            ["Facebook Page", str(row.get("Facebook Page",""))],
            ["Instagram", str(row.get("Instagram",""))],
            ["LinkedIn Company", str(row.get("LinkedIn Company URL",""))],
            ["YouTube", str(row.get("YouTube",""))],
            ["Twitter/X", str(row.get("Twitter",""))],
            ["Rating", str(row.get("Rating",""))],
            ["Audit", f"{row.get('Audit Score','')}/100 ({row.get('Audit Grade','')})"],
            ["Issues", str(row.get("Issues",""))],
            ["Pitch", str(row.get("Pitch Angle",""))],
        ]
        t = Table(d, colWidths=[1.4*inch, 5.5*inch])
        t.setStyle(TableStyle([("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                               ("FONTSIZE", (0,0), (-1,-1), 8.5),
                               ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                               ("TOPPADDING", (0,0), (-1,-1), 3),
                               ("LINEBELOW", (0,0), (-1,-2), 0.2, colors.HexColor("#EEEEEE"))]))
        st_.append(t); st_.append(Spacer(1, 0.2*inch))
    doc.build(st_); buf.seek(0)
    return buf.getvalue()


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Lead Generation Assistant", layout="wide", page_icon="🎯")

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

with st.sidebar:
    st.markdown("### 🎨 Theme")
    tc = st.radio("Theme", ["Dark","Light","System"],
                  index=["Dark","Light","System"].index(st.session_state["theme"]),
                  horizontal=True, label_visibility="collapsed")
    st.session_state["theme"] = tc
    if tc == "Light":
        st.markdown("""<style>
            .stApp { background-color: #fff; color: #111; }
            section[data-testid="stSidebar"] { background-color: #f4f4f9; }
            h1,h2,h3,h4,h5,h6,p,label,span { color: #111 !important; }
            </style>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🕘 Search History")
    hist = load_all_history()
    if not hist:
        st.caption("No history yet.")
    else:
        st.caption(f"{len(hist)} past searches")
        if st.button("🗑️ Clear All History", width='stretch', type="secondary"):
            clear_all_history(); st.success("Cleared!"); st.rerun()
        st.markdown("---")
        for item in hist[:30]:
            lbl = f"📌 {item['industry']} · {item['location']}"
            if item.get("keywords"): lbl += f" ({item['keywords']})"
            with st.expander(lbl):
                st.caption(f"🕐 {item['timestamp']}")
                st.caption(f"👥 {item.get('lead_count', 0)} leads")
                cA, cB = st.columns(2)
                with cA:
                    if st.button("📂 Load", key=f"l_{item['id']}", width='stretch'):
                        st.session_state["df"] = pd.DataFrame(item["leads"], columns=COLUMNS)
                        st.session_state["chat_history"] = []
                        st.rerun()
                with cB:
                    if st.button("🗑️ Delete", key=f"d_{item['id']}", width='stretch'):
                        delete_history(item["id"]); st.rerun()


st.title("🎯 Lead Generation Assistant")
st.caption("Multi-engine search · Executive finder · Email patterns · Clean Facebook/LinkedIn URLs · Smart country detection")

c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
with c1: industry = st.text_input("Industry", value="Med Spas")
with c2: location = st.text_input("Location", value="California")
with c3: num_leads = st.number_input("Number of Leads", min_value=1, max_value=50, value=10)
with c4: keywords = st.text_input("Keywords (optional)", placeholder="e.g. Botox, Laser")

if st.button("Find Leads", type="primary"):
    if not industry or not location:
        st.warning("Please fill in Industry and Location."); st.stop()
    ph_s, ph_e, ph_r = st.empty(), st.empty(), st.empty()
    prog = st.progress(0)
    ph_s.info("🔍 Searching (DuckDuckGo → Bing → Google)...")
    results = multi_search(industry, location, target=num_leads)
    if not results:
        st.error("No results. Try again in a few minutes."); st.stop()
    ph_s.success(f"✅ Collected {len(results)} pages.")
    prog.progress(15)
    ph_e.info(f"🧠 Extracting leads ({len(results)} pages)...")
    rows = extract_leads(industry, location, keywords, results, num_leads)
    if not rows:
        st.error("Could not extract leads."); st.stop()
    ph_e.success(f"✅ Extracted {len(rows)} leads.")
    prog.progress(25)
    df = clean_df(rows)
    er = df.to_dict(orient="records")
    total = len(er)
    for i, lead in enumerate(er):
        ph_r.info(f"🔬 Deep researching {i+1}/{total}: {lead.get('Business Name','')[:40]}")
        try:
            er[i] = enrich_lead_deep(lead)
        except Exception as e:
            st.warning(f"Enrich failed: {e}")
        prog.progress(25 + int(((i+1)/total) * 65))
    prog.empty(); ph_r.empty()
    df = clean_df(er)
    pi = build_previous_leads_index()
    df = mark_duplicates(df, pi)
    dc = (df["Duplicate"] == "Yes").sum()
    if dc > 0:
        st.info(f"🔁 {dc} duplicate(s) · {len(df)-dc} new")
    st.session_state["df"] = df
    st.session_state["chat_history"] = []
    st.session_state["search_industry"] = industry
    st.session_state["search_location"] = location
    st.session_state["search_keywords"] = keywords
    save_history(industry, location, keywords, df)


if "df" in st.session_state:
    df = st.session_state["df"]
    st.success(f"Found {len(df)} businesses")

    st.markdown("#### 🔎 Filter")
    fc, _ = st.columns([1, 3])
    with fc:
        vm = st.selectbox("Show:", ["All leads", "Only new (not seen before)", "Only duplicates"], key="filter_mode")
    vdf = df.copy()
    if vm == "Only new (not seen before)": vdf = vdf[vdf["Duplicate"] == "No"]
    elif vm == "Only duplicates": vdf = vdf[vdf["Duplicate"] == "Yes"]
    st.caption(f"Showing **{len(vdf)}** of **{len(df)}**")

    priority = ["Business Name", "Duplicate", "Executive Name", "Executive Title",
                "Executive Email", "Executive LinkedIn",
                "Lead Score", "Audit Grade", "Phone Number", "Email", "Website",
                "Street Address", "City", "State", "ZIP Code", "Country",
                "Google Maps URL", "Facebook Page", "Instagram", "LinkedIn Company URL",
                "YouTube", "Twitter"]
    priority = [c for c in priority if c in vdf.columns]
    others = [c for c in vdf.columns if c not in priority]
    st.dataframe(vdf[priority + others], width='stretch')

    st.markdown("---"); st.markdown("### 🔍 Website Quality Audit")
    if st.button("🚀 Run Website Audit", type="primary"):
        ap = st.progress(0); ast = st.empty()
        def upd(c, t, a):
            ap.progress(c/t); ast.info(f"🔍 {c}/{t}...")
        with st.spinner("Auditing..."):
            da = run_full_audit_on_df(df, cb=upd)
        ap.empty(); ast.success(f"✅ Audited!")
        st.session_state["df"] = da
        save_history(st.session_state.get("search_industry",""),
                     st.session_state.get("search_location",""),
                     st.session_state.get("search_keywords",""), da)
        st.rerun()

    st.markdown("---"); st.markdown("### 📧 Email Generator")
    ec1, ec2, ec3 = st.columns([3, 2, 2])
    with ec1:
        names = vdf["Business Name"].tolist() if not vdf.empty else df["Business Name"].tolist()
        sel_name = st.selectbox("Lead", names, key="email_lead_select")
    with ec2: sel_tpl = st.selectbox("Template", list(EMAIL_TEMPLATES.keys()), key="email_template_select")
    with ec3: sel_tone = st.selectbox("Tone", TONES, key="email_tone_select")
    if st.button("✨ Generate Email", type="primary"):
        row = df[df["Business Name"] == sel_name].iloc[0].to_dict()
        with st.spinner("Writing..."):
            st.session_state["generated_email"] = generate_email(sel_tpl, EMAIL_TEMPLATES[sel_tpl], sel_tone, row)
    if st.session_state.get("generated_email"):
        st.markdown("#### 📝 Generated Email")
        ed = st.text_area("Edit:", value=st.session_state["generated_email"], height=300)
        ch = f"""<button onclick="navigator.clipboard.writeText(`{ed.replace('`',"'").replace(chr(92), chr(92)*2)}`).then(()=>{{this.innerText='✅';setTimeout(()=>{{this.innerText='📋 Copy';}},2000);}})" style="width:100%;background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;">📋 Copy Email</button>"""
        components.html(ch, height=50)

    st.markdown("---"); st.markdown("### 📤 Export")
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    with ec1:
        st.download_button("📥 CSV", vdf.to_csv(index=False).encode(), "leads.csv", "text/csv", width='stretch')
    with ec2:
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                vdf.to_excel(w, index=False, sheet_name="Leads")
            st.download_button("📥 Excel", buf.getvalue(), "leads.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width='stretch')
        except Exception as e: st.warning(f"Excel: {e}")
    with ec3:
        with st.popover("📊 Sheets", width='stretch'):
            sn = st.text_input("Sheet Name", value="Lead Gen Results", key="gsheet_name")
            upl = st.file_uploader("JSON", type=["json"], key="gsheet_json")
            if st.button("🚀 Push Now", key="push_gsheet", width='stretch'):
                if upl is not None:
                    try:
                        c = json.load(upl)
                        with st.spinner("Pushing..."):
                            ok, m = push_to_google_sheets(vdf, sn, c)
                        st.success(m) if ok else st.error(m)
                    except Exception as e: st.error(f"JSON error: {e}")
                else: st.warning("Upload JSON.")
    with ec4:
        cs = vdf.to_csv(index=False).replace("`","'").replace("\\","\\\\")
        ch = f"""<button onclick="navigator.clipboard.writeText(`{cs}`).then(()=>{{this.innerText='✅';setTimeout(()=>{{this.innerText='📋 Copy';}},2000);}})" style="width:100%;background-color:#FF4B4B;color:white;padding:10px;border:none;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;">📋 Copy</button>"""
        components.html(ch, height=50)
    with ec5:
        if st.button("📄 PDF", width='stretch'):
            with st.spinner("Building..."):
                try:
                    st.session_state["pdf_bytes"] = generate_pdf_report(
                        vdf, st.session_state.get("search_industry",""),
                        st.session_state.get("search_location",""),
                        st.session_state.get("search_keywords",""))
                except Exception as e: st.error(f"PDF: {e}")
    if st.session_state.get("pdf_bytes"):
        st.download_button("⬇️ Download PDF", st.session_state["pdf_bytes"],
            f"lead_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            "application/pdf", width='stretch')

    st.markdown("---"); st.markdown("### 💬 Chat with Your Leads")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    uq = st.chat_input("Ask about these leads...")
    if uq:
        st.session_state["chat_history"].append({"role":"user","content":uq})
        with st.chat_message("user"): st.markdown(uq)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = ask_ai_about_leads(uq, df)
            st.markdown(ans)
        st.session_state["chat_history"].append({"role":"assistant","content":ans})