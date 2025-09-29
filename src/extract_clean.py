import os, csv, json, re, unicodedata, datetime, sys
from pathlib import Path

# Try PDF extractors
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAS_PDFMINER = True
except Exception:
    HAS_PDFMINER = False

from bs4 import BeautifulSoup

# project root (script in src/)
PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
CLEAN = PROJECT / "data" / "cleaned"
CHUNKS = PROJECT / "data" / "chunks"
RAW_META = PROJECT / "data" / "raw_meta"
EVID = RAW_META / "evidence"

# ensure dirs exist
for p in [RAW, CLEAN, CHUNKS, RAW_META, EVID]:
    p.mkdir(parents=True, exist_ok=True)

# chunk file (reset each run)
CHUNKFILE = CHUNKS / "chunks.jsonl"
if CHUNKFILE.exists():
    try:
        CHUNKFILE.unlink()
    except Exception as e:
        print("[WARN] Could not remove old chunk file:", e)

# basic clean patterns
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{6,})')

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'([A-Za-zÀ-ỹ])-\n([A-Za-zÀ-ỹ])', r'\1\2', s)
    s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = EMAIL_RE.sub("[EMAIL_REDACTED]", s)
    s = PHONE_RE.sub("[PHONE_REDACTED]", s)
    return s.strip()

def extract_html(path: Path) -> str:
    txt = path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(txt, "html.parser")
    main = soup.select_one("main") or soup
    return main.get_text("\n")

def extract_pdf_with_fitz(path: Path):
    doc = fitz.open(str(path))
    pages = []
    for p in doc:
        pages.append(p.get_text("text") or "")
    return "\n".join(pages), len(doc)

def extract_pdf_with_pdfminer(path: Path):
    try:
        text = pdfminer_extract_text(str(path))
        # pdfminer does not return page count easily; set 0
        return text, 0
    except Exception:
        return "", 0

def extract_pdf(path: Path):
    if HAS_FITZ:
        try:
            return extract_pdf_with_fitz(path)
        except Exception as e:
            print(f"[WARN] fitz failed for {path.name}: {e}")
            # fallback to pdfminer if available
    if HAS_PDFMINER:
        try:
            return extract_pdf_with_pdfminer(path)
        except Exception as e:
            print(f"[WARN] pdfminer failed for {path.name}: {e}")
    # last resort: try reading as text (may return binary header)
    try:
        raw = path.read_text(encoding='utf-8', errors='ignore')
        return raw, 0
    except Exception as e:
        print(f"[WARN] raw read failed for {path.name}: {e}")
        return "", 0

def is_scan_pdf(text: str, pages: int) -> bool:
    # heuristic: if no pages info, but starts with '%PDF-' or very low chars => mark as needs_ocr
    if pages <= 0:
        if text.strip().startswith("%PDF-"):
            return True
        # if very short text, mark for OCR
        return len(text.strip()) < 500
    avg = len(text) / max(1, pages)
    return avg < 120

def safe_doc_id_from_path(p: Path) -> str:
    # create safe id (no spaces, only ascii-friendly)
    name = p.stem
    safe = re.sub(r'\s+', '_', name)
    safe = re.sub(r'[^\w\-_\.]', '', safe)
    return safe[:200]  # truncate if too long

# iterate raw files (only pdf/html/txt)
catalog = []
processed = 0
for f in sorted(RAW.iterdir()):
    if not f.is_file(): 
        continue
    suffix = f.suffix.lower()
    if suffix not in (".pdf", ".html", ".htm", ".txt"):
        print(f"[SKIP] Unsupported file type: {f.name}")
        continue

    doc_id = safe_doc_id_from_path(f)
    rec = {
        "id": doc_id,
        "local_path": str(f),
        "url": "",
        "harvested_at": datetime.datetime.utcnow().isoformat() + "Z",
        "needs_ocr": "false"
    }
    txt = ""
    pages = 0
    print(f"[INFO] Processing {f.name} ...")
    try:
        if suffix in (".html", ".htm"):
            txt = extract_html(f)
        elif suffix == ".pdf":
            txt, pages = extract_pdf(f)
            if is_scan_pdf(txt, pages):
                rec["needs_ocr"] = "true"
                print(f"[INFO] -> Marked needs_ocr for {f.name} (pages={pages}, chars={len(txt)})")
        else:  # .txt fallback
            txt = f.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"[ERROR] Exception while extracting {f.name}: {e}")
        txt = ""

    if not txt or len(txt.strip()) == 0:
        print(f"[WARN] No text extracted for {f.name}. If it's a scanned PDF, run OCR and save to cleaned file.")
        # still append record (so you know it's present)
        rec["cleaned_path"] = ""
        rec["cleaned_at"] = ""
    else:
        cleaned = clean_text(txt)
        out = CLEAN / f"{doc_id}.clean.txt"
        try:
            out.write_text(cleaned, encoding='utf-8')
            rec["cleaned_path"] = str(out)
            rec["cleaned_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            # chunking (char-based) - overwrite existing chunks file behavior handled earlier
            size, overlap = 900, 180
            i = 0; L = len(cleaned)
            with open(CHUNKFILE, "a", encoding='utf-8') as ch:
                while i < L:
                    end = min(i + size, L)
                    chunk = cleaned[i:end]
                    rec_chunk = {
                        "id": f"{doc_id}_chunk_{i}",
                        "doc_id": doc_id,
                        "text": chunk,
                        "harvested_at": rec["cleaned_at"]
                    }
                    ch.write(json.dumps(rec_chunk, ensure_ascii=False) + "\n")
                    i = max(i + size - overlap, end)
            print(f"[OK] Cleaned saved: {out.name}; chunks appended.")
            processed += 1
        except Exception as e:
            print(f"[ERROR] Writing cleaned file failed for {f.name}: {e}")
            rec["cleaned_path"] = ""
            rec["cleaned_at"] = ""

    catalog.append(rec)

# write minimal catalog_enriched.csv
keys = ["id", "local_path", "cleaned_path", "cleaned_at", "needs_ocr", "harvested_at"]
out_catalog = RAW_META / "catalog_enriched.csv"
try:
    with open(out_catalog, "w", encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in catalog:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"[DONE] catalog written: {out_catalog}")
except Exception as e:
    print("[ERROR] Writing catalog_enriched.csv failed:", e)

print(f"[SUMMARY] processed cleaned files: {processed} / {len(catalog)} (total raw present: {len(list(RAW.iterdir()))})")
print(" - cleaned folder:", CLEAN)
print(" - chunks file:", CHUNKFILE)
print(" - catalog enriched:", out_catalog)
