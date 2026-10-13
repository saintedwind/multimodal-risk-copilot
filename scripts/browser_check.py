"""Isolated Chromium dashboard checks; no user-browser automation."""

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)
env["REPORT_PROVIDER"] = "deterministic"
with (ROOT / "reports/browser.log").open("w") as log:
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/app/dashboard.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=log,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline or p.poll() is not None:
                    raise RuntimeError("Dashboard unavailable")
                time.sleep(0.5)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000})
            page.goto("http://127.0.0.1:8501")
            page.get_by_role("heading", name="Market Risk Review Desk").wait_for()
            page.get_by_text("Daily loss VaR / 95%", exact=True).wait_for(timeout=60000)
            page.get_by_role("tab", name="Scenario assumptions").click()
            page.get_by_text("GBM fitted to log returns;", exact=False).wait_for()
            page.get_by_role(
                "heading", name="One-step risk evaluation", exact=True
            ).wait_for()
            page.locator('[data-testid="stDataFrame"]:visible').first.wait_for()
            page.screenshot(
                path=str(ROOT / "reports/desk-risk-evaluation.png"), full_page=True
            )
            page.get_by_role("tab", name="Price evidence").click()
            page.get_by_text("2747", exact=True).wait_for()
            page.screenshot(path=str(ROOT / "reports/desk-desktop.png"), full_page=True)
            page.get_by_role("tab", name="Review notes").click()
            page.get_by_text("Report mode: deterministic", exact=True).wait_for()
            mobile = b.new_page(viewport={"width": 390, "height": 844})
            mobile.goto("http://127.0.0.1:8501")
            title = mobile.get_by_role("heading", name="Market Risk Review Desk")
            title.wait_for()
            mobile.get_by_text("Daily loss VaR / 95%", exact=True).wait_for(
                timeout=60000
            )
            assert title.evaluate(
                "(e)=>{const r=e.getBoundingClientRect();const p=document.elementFromPoint(r.x+r.width/2,r.y+20);return e===p||e.contains(p)}"
            ), "Mobile headline is obscured"
            assert mobile.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), "Mobile overflow"
            mobile.screenshot(
                path=str(ROOT / "reports/desk-mobile.png"), full_page=True
            )
            mobile.get_by_role("tab", name="Review notes").click()
            mobile.get_by_text("Report mode: deterministic", exact=True).wait_for()
            assert not mobile.locator('[data-testid="stException"]').count()
            assert not page.locator('[data-testid="stException"]').count()
            b.close()
            print(
                "Chromium desktop/mobile, numeric results, assumptions and report passed"
            )
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
