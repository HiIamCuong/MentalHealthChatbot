# src/chunk_tokenwise.py
"""
Token-aware chunking using tiktoken.
Outputs:
 - project/data/chunks_token/chunks.jsonl  (one JSON per chunk)
 - project/data/chunk_meta/chunk_metadata.csv  (summary metadata)
Safe against missing raw files; won't try to open directories.
"""
import json, csv, datetime, sys
from pathlib import Path
from hashlib import sha256

try:
    import tiktoken
except Exception as e:
    print("Missing dependency tiktoken. Install: pip install tiktoken")
    raise

try:
    from tqdm import tqdm
    USE_TQDM = True
except Exception:
    USE_TQDM = False

PROJECT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT / "data" / "cleaned"
OUT_DIR = PROJECT / "data" / "chunks_token"
META_DIR = PROJECT / "data" / "chunk_meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

OUT_CHUNKS = OUT_DIR / "chunks.jsonl"
META_CSV = META_DIR / "chunk_metadata.csv"

# Tokenizer: cl100k_base works for gpt-3.5/gpt-4 family encodings
ENC = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 800
OVERLAP_TOKENS = 100

def file_sha256(path: Path) -> str:
    """Return hex sha256 for a regular file; return empty string otherwise."""
    try:
        if not path or not path.exists() or not path.is_file():
            return ""
        h = sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(65536), b""):
                h.update(b)
        return h.hexdigest()
    except Exception as e:
        # don't crash the whole pipeline for checksum issues
        print(f"[WARN] sha256 failed for {path}: {e}", file=sys.stderr)
        return ""

def build_token_char_mapping(text: str):
    """
    Best-effort mapping from token index -> (start_char, end_char).
    Returns (token_ids, token_offsets) where token_offsets is list of (start_char,end_char).
    This is best-effort and may be imperfect for some inputs.
    """
    token_ids = ENC.encode(text)
    token_offsets = []
    pos_search = 0
    for tok in token_ids:
        tok_text = ENC.decode([tok])
        if not tok_text:
            # safety: skip empty decoded token
            token_offsets.append((pos_search, pos_search))
            continue
        # find tok_text in original text starting at pos_search
        idx = text.find(tok_text, pos_search)
        if idx == -1:
            # if not found, try global find (may match earlier instance)
            idx = text.find(tok_text)
        if idx == -1:
            # fallback: assume it starts at pos_search
            idx = pos_search
        token_offsets.append((idx, idx + len(tok_text)))
        pos_search = idx + len(tok_text)
    return token_ids, token_offsets

def chunk_token_windows(token_ids, max_tokens, overlap):
    i = 0
    L = len(token_ids)
    while i < L:
        end = min(i + max_tokens, L)
        yield i, end, token_ids[i:end]
        if end == L:
            break
        i = max(i + max_tokens - overlap, end)

def find_raw_candidate(doc_stem: str):
    """Return first existing raw file Path (pdf/html/htm) or None."""
    raw_dir = PROJECT / "data" / "raw"
    for ext in [".pdf", ".PDF", ".html", ".htm", ".HTML", ".HTM"]:
        cand = raw_dir / (doc_stem + ext)
        if cand.is_file():
            return cand
    return None

def main():
    # overwrite outputs each run
    if OUT_CHUNKS.exists():
        OUT_CHUNKS.unlink()
    if META_CSV.exists():
        META_CSV.unlink()

    files = sorted(CLEAN_DIR.glob("*.clean.txt"))
    if not files:
        print(f"[WARN] No cleaned files found in {CLEAN_DIR}. Run Day3 extract first.")
        return

    iterator = tqdm(files, desc="Chunking files") if USE_TQDM else files

    with open(OUT_CHUNKS, "w", encoding="utf-8") as out_ch, open(META_CSV, "w", encoding="utf-8", newline='') as meta_f:
        csv_writer = csv.writer(meta_f)
        csv_writer.writerow(["doc_id","chunk_id","start_char","end_char","start_token","end_token","sha256","local_path","harvested_at","created_at","language","license"])
        total_chunks = 0
        for p in iterator:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"[WARN] cannot read cleaned file {p}: {e}", file=sys.stderr)
                continue
            doc_id = p.stem
            raw_candidate = find_raw_candidate(p.stem)
            sha = file_sha256(raw_candidate) if raw_candidate is not None else ""
            harvested_at = ""  # option: read from catalog_enriched.csv if present
            # CORRECT created_at: use datetime.datetime.utcnow()
            created_at = datetime.datetime.utcnow().isoformat() + "Z"

            try:
                token_ids, token_offsets = build_token_char_mapping(text)
            except Exception as e:
                print(f"[WARN] tokenization fail for {p}: {e}", file=sys.stderr)
                token_ids, token_offsets = [], []

            if not token_ids:
                # nothing to chunk — still write zero-chunk record? we skip
                continue

            for t_start, t_end, chunk_ids in chunk_token_windows(token_ids, TARGET_TOKENS, OVERLAP_TOKENS):
                try:
                    chunk_text = ENC.decode(chunk_ids)
                except Exception:
                    # fallback: try char-slice using token_offsets (if available)
                    if token_offsets and 0 <= t_start < len(token_offsets):
                        start_char = token_offsets[t_start][0]
                        end_char = token_offsets[t_end-1][1] if (t_end-1) < len(token_offsets) else start_char + 1
                        chunk_text = text[start_char:end_char]
                    else:
                        chunk_text = ""

                # determine char offsets using token_offsets (best-effort)
                if token_offsets and 0 <= t_start < len(token_offsets) and (t_end-1) < len(token_offsets):
                    start_char = token_offsets[t_start][0]
                    end_char = token_offsets[t_end-1][1]
                elif token_offsets and 0 <= t_start < len(token_offsets):
                    start_char = token_offsets[t_start][0]
                    end_char = start_char + len(chunk_text)
                else:
                    start_char = 0
                    end_char = start_char + len(chunk_text)

                chunk_id = f"{doc_id}_tok_{t_start}_{t_end}"
                rec = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_token": t_start,
                    "end_token": t_end,
                    "sha256": sha,
                    "local_path": str(p),
                    "harvested_at": harvested_at,
                    "created_at": created_at,
                    "language": "",
                    "license": ""
                }
                out_ch.write(json.dumps(rec, ensure_ascii=False) + "\n")
                csv_writer.writerow([doc_id, chunk_id, start_char, end_char, t_start, t_end, sha, str(p), harvested_at, created_at, "", ""])
                total_chunks += 1
    print(f"Done. wrote {total_chunks} chunks to: {OUT_CHUNKS}")
    print("Metadata CSV:", META_CSV)

if __name__ == "__main__":
    main()
