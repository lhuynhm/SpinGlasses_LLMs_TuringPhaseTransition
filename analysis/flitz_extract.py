# -*- coding: utf-8 -*-
"""
Flitz survey (2026) step 2 of 3 — extract flitz text from the uploaded files.

Reads every submission in Data for Figures/flitz_pdfs/ and recovers the text the
*sender* actually wrote, so the result is comparable with the original 33-flitz
human corpus (human_written_flitzes.xlsx), which holds clean flitz bodies only.

Three source types occur in the 2026 uploads:
  * .pdf with a text layer  -> PyMuPDF direct extraction (most uploads)
  * .eml (raw email)        -> text/plain part
  * .pdf that is a screenshot with NO text layer -> needs OCR. Put a manual
    transcription at flitz_pdfs/ocr/<same stem>.txt and it is picked up here
    (source="ocr_manual"). Nothing is invented: without the sidecar such a file
    is reported has_text=False, needs_ocr=True.

Cleaning (why it matters): most uploads are Gmail *print-to-PDF* files, so the
raw text layer carries chrome that was never part of the flitz — account/sender
headers, "To: ...", the "8/7/26, 7:48 PM / Dartmouth College Mail - <subject> /
https://mail.google.com/... / 1/3" print footer repeated per page, and, in
threads, the RECIPIENT'S REPLY and later messages. That boilerplate is shared
across documents, so leaving it in would inflate the pairwise cosine. We keep
only the first message's body and drop the chrome; emoji, gif alt-text-free
prose and sign-offs are kept, matching the baseline corpus convention.

File -> respondent mapping: uploads were downloaded manually and are named after
the sender, not the response row, so the old "digits in filename" rule is wrong.
Instead each file's mtime (= Drive creation time = submission time) is matched to
the manifest Timestamp within --match_tol_s. The resolved map is cached to
results/flitz_file_map.csv and reused on later runs (mtimes do not survive a
re-copy). Matching is skipped gracefully if the manifest is absent.

Outputs
    Data for Figures/flitz_survey2026_texts.xlsx  - one flitz per row (header=None),
                                                    ready for flitz_similarity.py
    results/flitz_extracted.csv                   - per-file text, char counts,
                                                    has_text/needs_ocr, + metadata
    results/flitz_file_map.csv                    - file <-> respondent_id map

Run:
    ../.venv/bin/python flitz_extract.py
"""
from __future__ import annotations

import argparse
import email
import email.policy
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf

from common import DATA_DIR, DEFAULT_RESULTS, ensure_dir

PDF_DIR = DATA_DIR / "flitz_pdfs"
TEXTS_XLSX = DATA_DIR / "flitz_survey2026_texts.xlsx"

# --- Gmail print-to-PDF chrome -------------------------------------------------
DATE_LINE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\w{3}\s+\d{1,2},\s+\d{4}\s+at\s+"
    r"\d{1,2}:\d{2}\s*[AP]M\s*$")
FROM_LINE = re.compile(r"^.{1,80}<[^>]+@[^>]+>\s*$")          # "Name <a@b.edu>"
ADDR_LINE = re.compile(r"^(Draft\s+)?(To|Cc|Bcc|Reply-To):", re.I)
COUNT_LINE = re.compile(r"^\d+\s+messages?\s*$", re.I)
PRINT_TS = re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*[AP]M\s*$")
MAIL_URL = re.compile(r"^https?://mail\.google\.com/\S*$")
PAGE_NO = re.compile(r"^\d+/\d+\s*$")
QUOTED = re.compile(r"^\[Quoted text hidden\]\s*$", re.I)
IMG_ALT = re.compile(r"\[image:.*?\]", re.S)                   # gmail plain-text gif alt
SIG_SEP = re.compile(r"^-{2}\s*$")                             # "-- " signature block
GIF_FILE = re.compile(r"^\S+\.(gif|png|jpe?g)\s*$", re.I)      # bare "rUwEVz.gif"


def _norm(text: str) -> str:
    """NFKC-fold styled unicode (𝕍→V, ｆ→f, …), tidy whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_print_chrome(lines):
    """Drop the per-page print footer, which Gmail emits as a fixed 4-line block:

        8/7/26, 7:48 PM
        Dartmouth College Mail - <subject>
        https://mail.google.com/mail/u/0/?ik=...
        1/3
    """
    out, i, n = [], 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if PRINT_TS.match(s):
            j = i + 1
            while j < min(i + 5, n) and not MAIL_URL.match(lines[j].strip()):
                j += 1
            if j < min(i + 5, n):                       # footer block confirmed
                i = j + 1
                if i < n and PAGE_NO.match(lines[i].strip()):
                    i += 1
                continue
        if MAIL_URL.match(s) or PAGE_NO.match(s) or COUNT_LINE.match(s):
            i += 1
            continue
        if QUOTED.match(s) or GIF_FILE.match(s):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def first_message_body(text: str) -> str:
    """Keep only the first message of a Gmail thread print; drop headers.

    Google-Docs prints (no email headers at all) pass through untouched.
    """
    lines = text.split("\n")
    date_idx = [i for i, ln in enumerate(lines) if DATE_LINE.match(ln.strip())]
    if not date_idx:
        return "\n".join(_strip_print_chrome(lines))

    start = date_idx[0] + 1
    while start < len(lines) and (ADDR_LINE.match(lines[start].strip())
                                  or not lines[start].strip()):
        start += 1
    # body ends just before the sender line of the next message in the thread
    if len(date_idx) > 1:
        end = date_idx[1]
        while end > start and (not lines[end - 1].strip()
                               or FROM_LINE.match(lines[end - 1].strip())):
            end -= 1
    else:
        end = len(lines)

    return "\n".join(_strip_print_chrome(lines[start:end]))


def extract_pdf(path: Path) -> str:
    try:
        doc = pymupdf.open(path)
    except Exception as e:                                   # pragma: no cover
        print(f"   ! cannot open {path.name}: {e}")
        return ""
    raw = "\n".join(page.get_text() for page in doc)
    doc.close()
    return _norm(first_message_body(_norm(raw)))


def extract_eml(path: Path) -> str:
    with open(path, "rb") as fh:
        msg = email.message_from_binary_file(fh, policy=email.policy.default)
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    body = part.get_content()
    if part.get_content_type() == "text/html":
        body = re.sub(r"<[^>]+>", " ", body)
    body = IMG_ALT.sub(" ", body)
    lines = []
    for ln in body.split("\n"):
        if SIG_SEP.match(ln.strip()):          # cut the email signature block
            break
        lines.append(ln)
    body = "\n".join(lines).replace("*", "")
    return _norm(body)


def build_file_map(files, manifest: pd.DataFrame, tol_s: int):
    """Match each upload to a response row by file mtime <-> response Timestamp.

    The mtimes are the Drive files' creation times (= submission times), but the
    manual download wrote them with a constant timezone shift relative to the
    Forms Timestamp column, so absolute times cannot be compared directly. Both
    series are strictly increasing, so we pair them in time order, estimate that
    single offset as the median gap, and require every residual to fall inside
    --match_tol_s; otherwise the pairing is reported unmatched rather than
    guessed.
    """
    ts = pd.to_datetime(manifest["timestamp"], format="mixed", utc=True).sort_values()
    mt = pd.Series({f.name: pd.Timestamp(datetime.fromtimestamp(f.stat().st_mtime,
                                                               tz=timezone.utc))
                    for f in files}).sort_values()
    if len(ts) != len(mt):
        print(f"   ! {len(mt)} uploads vs {len(ts)} responses — cannot pair by time; "
              f"respondent_id left blank.")
        return pd.DataFrame({"file": [f.name for f in files],
                             "respondent_id": None, "match_residual_s": None})

    gaps = (mt.to_numpy() - ts.to_numpy()).astype("timedelta64[s]").astype(float)
    offset = float(pd.Series(gaps).median())
    resid = gaps - offset
    print(f"[flitz] file/response time offset {offset / 3600:.2f} h; "
          f"max residual {abs(resid).max():.0f} s")

    rows = []
    for fname, rid, r in zip(mt.index, manifest.loc[ts.index, "respondent_id"], resid):
        ok = abs(r) <= tol_s
        rows.append({"file": fname,
                     "respondent_id": int(rid) if ok else None,
                     "match_residual_s": round(float(r), 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default=str(PDF_DIR))
    ap.add_argument("--out_texts", default=str(TEXTS_XLSX))
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--min_chars", type=int, default=20,
                    help="Minimum extracted characters to count as a real flitz.")
    ap.add_argument("--match_tol_s", type=int, default=120,
                    help="Max |file mtime - response timestamp| for a respondent match.")
    ap.add_argument("--remap", action="store_true",
                    help="Recompute the file<->respondent map instead of reusing the cache.")
    ap.add_argument("--ignore_transcriptions", action="store_true",
                    help="Ignore flitz_pdfs/transcriptions/*.txt (machine-extractable text only).")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    files = sorted([p for p in pdf_dir.iterdir()
                    if p.suffix.lower() in (".pdf", ".eml") and not p.name.startswith("~$")])
    if not files:
        raise SystemExit(
            f"No .pdf/.eml uploads found in {pdf_dir}. Run flitz_download.py first, or "
            f"drop the downloaded Drive folder's files into that directory.")

    trans_dir = pdf_dir / "transcriptions"
    out_dir = ensure_dir(Path(args.out_dir))

    rows = []
    for f in files:
        is_eml = f.suffix.lower() == ".eml"
        text = extract_eml(f) if is_eml else extract_pdf(f)
        source = "eml" if is_eml else "pdf_text_layer"

        # image-only PDF (a screenshot of the flitz): no text layer to extract
        needs_ocr = False
        if not is_eml and len(text) < args.min_chars:
            with pymupdf.open(f) as doc:
                needs_ocr = any(page.get_images(full=True) for page in doc)

        sidecar = trans_dir / (f.stem + ".txt")
        if sidecar.exists() and not args.ignore_transcriptions:
            text = _norm(sidecar.read_text(encoding="utf-8"))
            source = "manual_transcription"

        rows.append({
            "file": f.name,
            "source": source,
            "needs_ocr": needs_ocr,
            "char_count": len(text),
            "has_text": len(text) >= args.min_chars,
            "text": text,
        })
    ext = pd.DataFrame(rows)

    # ---- respondent mapping (cached) ----
    map_path = out_dir / "flitz_file_map.csv"
    man_path = out_dir / "flitz_manifest.csv"
    fmap = None
    if map_path.exists() and not args.remap:
        fmap = pd.read_csv(map_path)
        if set(fmap["file"]) != set(ext["file"]):
            fmap = None                                   # stale cache -> rebuild
    if fmap is None and man_path.exists():
        fmap = build_file_map(files, pd.read_csv(man_path), args.match_tol_s)
        fmap.to_csv(map_path, index=False)
    if fmap is not None:
        fmap["respondent_id"] = pd.to_numeric(fmap["respondent_id"], errors="coerce").astype("Int64")
        ext = ext.merge(fmap, on="file", how="left")
        if man_path.exists():
            meta = pd.read_csv(man_path)
            meta["respondent_id"] = meta["respondent_id"].astype("Int64")
            ext = ext.merge(meta, on="respondent_id", how="left")

    # Rows were built in upload-filename order (files are named after the sender),
    # which is arbitrary. Emit in respondent order instead — i.e. submission order —
    # so flitz_extracted.csv and the texts xlsx share one stable, meaningful
    # ordering and downstream labels (r0, r3, r4, ...) come out sorted.
    # Unmatched files (no respondent_id) sort last, keeping filename order.
    if "respondent_id" in ext.columns:
        ext = (ext.sort_values("respondent_id", na_position="last", kind="stable")
               .reset_index(drop=True))

    ext.to_csv(out_dir / "flitz_extracted.csv", index=False)

    kept = ext[ext["has_text"]].copy()
    # one flitz per row, no header (matches human_written_flitzes.xlsx convention)
    kept[["text"]].to_excel(args.out_texts, index=False, header=False)

    n_ocr_missing = int((ext["needs_ocr"] & ~ext["has_text"]).sum())
    print(f"[flitz] {len(ext)} uploads, {kept.shape[0]} with usable text "
          f"(>= {args.min_chars} chars), {len(ext) - kept.shape[0]} blank/short"
          + (f", {n_ocr_missing} image-only awaiting OCR" if n_ocr_missing else "") + ".")
    print(f"[flitz] wrote {args.out_texts}")
    print(f"[flitz] wrote {out_dir / 'flitz_extracted.csv'}")
    cols = [c for c in ["file", "respondent_id", "source", "char_count", "has_text",
                        "needs_ocr", "ai_pct", "outcome"] if c in ext.columns]
    with pd.option_context("display.max_colwidth", 46, "display.width", 200):
        print(ext[cols].to_string(index=False))


if __name__ == "__main__":
    main()
