"""Render reports/report.md to reports/AML_Dengue_Report.pdf.

Requires pandoc (https://pandoc.org) and Playwright's Chromium
(`pip install playwright && playwright install chromium`).
"""
import subprocess
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main():
    html = REPORTS / "report.html"
    subprocess.run(["pandoc", "report.md", "-s", "--mathml", "--css", "report.css",
                    "--embed-resources", "-o", html.name], cwd=REPORTS, check=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(f"Playwright not installed - open {html} in a browser and print to PDF instead.")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html.as_uri())
        page.pdf(path=str(REPORTS / "AML_Dengue_Report.pdf"), format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#8a8984">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
        browser.close()
    html.unlink()
    print("Wrote", REPORTS / "AML_Dengue_Report.pdf")


if __name__ == "__main__":
    main()
