"""Build the report PDF from report/report.typ (Typst) and report its page count.

    python report/build.py
"""
from pathlib import Path

import typst
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
pdf = HERE / "report.pdf"
typst.compile(str(HERE / "report.typ"), output=str(pdf), root=str(HERE))
print(f"{pdf}: {len(PdfReader(pdf).pages)} pages")
