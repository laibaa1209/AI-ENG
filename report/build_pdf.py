"""report.md -> report.html -> report.pdf (headless Edge/Chrome).

    python report/build_pdf.py
"""
import os
import re
import shutil
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """body{font-family:Georgia,serif;max-width:820px;margin:auto;padding:0 16px;line-height:1.5;font-size:11pt;color:#111}
h1{font-size:22pt}h2{border-bottom:1px solid #ccc;margin-top:1.6em}img{max-width:100%;display:block;margin:12px auto 2px}
em{color:#333}table{border-collapse:collapse;font-size:8.5pt;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 6px}th{background:#eee}
pre{background:#f5f5f5;padding:8px;font-size:8.5pt;white-space:pre-wrap;page-break-inside:avoid}code{font-size:90%}"""


def main():
    md = open(os.path.join(HERE, "report.md"), encoding="utf-8").read()
    maths = []

    def stash(m):  # keep $$...$$ away from the markdown parser
        maths.append(m.group(0))
        return f"MATHBLOCK{len(maths) - 1}X"

    md = re.sub(r"\$\$.*?\$\$", stash, md, flags=re.S)
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    for i, m in enumerate(maths):
        body = body.replace(f"MATHBLOCK{i}X", m)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>MiniGPT report</title><style>{CSS}</style>
<script>MathJax={{tex:{{displayMath:[['$$','$$']],inlineMath:[]}}}}</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script></head><body>{body}</body></html>"""
    html_path = os.path.join(HERE, "report.html")
    open(html_path, "w", encoding="utf-8").write(html)

    browser = next((b for b in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                shutil.which("chromium"), shutil.which("google-chrome")] if b and os.path.exists(b)), None)
    if not browser:
        print(f"wrote {html_path}; no Edge/Chrome found, print it to PDF from a browser")
        return
    pdf = os.path.join(HERE, "report.pdf")
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf}", "file:///" + html_path.replace("\\", "/")], check=True)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
