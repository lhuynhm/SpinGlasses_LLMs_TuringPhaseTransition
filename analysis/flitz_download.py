# -*- coding: utf-8 -*-
"""
Flitz survey (2026) step 1 of 3 — build a manifest and fetch the PDF uploads.

The 'Flizting Survey2026.csv' upload column holds Google Drive LINKS, not files,
e.g. https://drive.google.com/u/0/open?usp=forms_web&id=<FILE_ID>. This script:
  * parses the file IDs + all response metadata into results/flitz_manifest.csv
  * tries to download each PDF into Data for Figures/flitz_pdfs/<respondent>.pdf

Download caveat: Google-Forms upload files live in a Drive folder you (the form
owner) control and are usually NOT link-shared, so automated download often hits
a permission wall. Two reliable options if gdown fails:
  (A) Open the form's Drive upload folder in a browser, "Download all" -> unzip
      into Data for Figures/flitz_pdfs/. Name doesn't matter; step 2 globs *.pdf.
  (B) Temporarily set the folder to "Anyone with the link", then re-run this.

Run:
    .venv/bin/python analysis/flitz_download.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from common import DATA_DIR, DEFAULT_RESULTS, ensure_dir

DEFAULT_FLITZ_CSV = DATA_DIR / "Flizting Survey2026.csv"
PDF_DIR = DATA_DIR / "flitz_pdfs"

UPLOAD_PREFIX = "Please upload the most recent flitz"
# map the (long) Forms question headers to short manifest column names
META = {
    "Timestamp": "timestamp",
    "Who did you get this survey from?": "recruited_by",
    "On which date did you send the flitz you have just uploaded?": "date_sent",
    "How much percentage of your flitz was written by AI?": "ai_pct",
    "What was the outcome of your flitz?": "outcome",
    "How did you decide to flitz this recipient?": "decided_how",
    "What is your ethnicity?": "ethnicity",
    "What did you think your flitz recipient's ethnicity was?": "recipient_ethnicity",
}


def drive_id(url: str):
    if not isinstance(url, str):
        return None
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url) or re.search(r"/d/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flitz_csv", default=str(DEFAULT_FLITZ_CSV))
    ap.add_argument("--pdf_dir", default=str(PDF_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--no_download", action="store_true",
                    help="Only write the manifest; skip the gdown attempt.")
    args = ap.parse_args()

    df = pd.read_csv(args.flitz_csv)
    upload_col = next(c for c in df.columns
                      if c.startswith(UPLOAD_PREFIX) and not c.rstrip().endswith(("[Score]", "[Feedback]")))

    rows = []
    for idx, r in df.iterrows():
        rec = {"respondent_id": idx}
        for src, dst in META.items():
            rec[dst] = r[src] if src in df.columns else None
        rec["drive_url"] = r[upload_col]
        rec["drive_id"] = drive_id(r[upload_col])
        rows.append(rec)
    manifest = pd.DataFrame(rows)

    out_dir = ensure_dir(Path(args.out_dir))
    man_path = out_dir / "flitz_manifest.csv"
    manifest.to_csv(man_path, index=False)
    print(f"[flitz] wrote {man_path}  ({manifest['drive_id'].notna().sum()}/{len(manifest)} have a Drive ID)")

    if args.no_download:
        return

    pdf_dir = ensure_dir(Path(args.pdf_dir))
    import gdown
    ok, fail = 0, 0
    for _, r in manifest.iterrows():
        if not r["drive_id"]:
            continue
        out = pdf_dir / f"respondent_{int(r['respondent_id']):02d}.pdf"
        if out.exists():
            ok += 1
            continue
        try:
            got = gdown.download(id=r["drive_id"], output=str(out), quiet=True)
            if got:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"   ! respondent {r['respondent_id']}: {e}")
    print(f"[flitz] downloaded {ok} PDF(s) to {pdf_dir}; {fail} failed.")
    if fail:
        print("[flitz] Some downloads failed (likely Drive permissions). See the "
              "download caveat at the top of this file — download the folder "
              "manually into the pdf_dir, then run flitz_extract.py.")


if __name__ == "__main__":
    main()
