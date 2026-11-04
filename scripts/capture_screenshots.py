from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

PORT = 8511
URL = f"http://localhost:{PORT}/?preload_validation=1"
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
PORTFOLIO_DIR = Path(r"C:\Projects\Portfolio-site\assets\images\projects\algorithmic-trading-backtester")
MARKET_DATA = ROOT / "reports" / "market_data_2019_2026.csv"
MANIFEST_PATH = ROOT / "reports" / "screenshot_data_manifest.json"
BASELINE_STRATEGY = "Moving Average Crossover"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url, timeout=35):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def reset_scrolls(page):
    page.evaluate("window.scrollTo(0, 0)")
    page.evaluate("""
        const sb = document.querySelector('section[data-testid="stSidebar"]');
        if (sb) sb.scrollTop = 0;
        const sbc = document.querySelector('[data-testid="stSidebarContent"]');
        if (sbc) sbc.scrollTop = 0;
    """)
    page.wait_for_timeout(300)


def save_screenshot_formats(raw_png_path, stem):
    docs_png = DOCS_DIR / f"{stem}.png"
    portfolio_webp = PORTFOLIO_DIR / f"{stem}.webp"
    with Image.open(raw_png_path) as source:
        source.load()
        for attempt in range(5):
            try:
                source.save(str(docs_png), format="PNG", optimize=True)
                source.save(str(portfolio_webp), format="WEBP", quality=92, method=6)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.75)
        image_size = source.size
    print(f"Saved {stem}: {docs_png} and {portfolio_webp} ({image_size})")


def verify_market_data():
    if not MARKET_DATA.exists():
        raise FileNotFoundError(
            f"Missing cached market data: {MARKET_DATA}. "
            "Download the legitimate Yahoo Finance cache before capturing screenshots."
        )
    import pandas as pd

    frame = pd.read_csv(MARKET_DATA, parse_dates=["Date"])
    frame = frame.set_index("Date").sort_index()
    effective_start = frame.index.min().date().isoformat()
    effective_end = frame.index.max().date().isoformat()
    years = (frame.index.max() - frame.index.min()).days / 365.25
    if years < 5 or len(frame) < 1000:
        raise ValueError(f"Cached market data is not multi-year: {len(frame)} rows, {years:.1f} years")
    print(
        f"Using cached market data: {MARKET_DATA} | "
        f"rows={len(frame)} | effective={effective_start} to {effective_end}"
    )
    return {
        "source": "Yahoo Finance via yfinance",
        "local_path": str(MARKET_DATA),
        "columns": list(frame.columns),
        "rows": len(frame),
        "effective_start": effective_start,
        "effective_end": effective_end,
    }


def open_sidebar(page):
    button = page.locator('button[data-testid="stSidebarCollapseButton"]')
    if button.count() and not page.locator('section[data-testid="stSidebar"]').is_visible():
        button.first.click()
        page.wait_for_timeout(500)


def close_sidebar(page):
    button = page.locator('button[data-testid="stSidebarCollapseButton"]')
    if button.count() and page.locator('section[data-testid="stSidebar"]').is_visible():
        button.first.click()
        page.wait_for_timeout(500)


def prepare_view(page, tab_name, requires_plot=True):
    tab = page.get_by_role("tab", name=tab_name, exact=True)
    tab.wait_for(state="visible", timeout=15000)
    tab.click()
    reset_scrolls(page)
    if requires_plot:
        page.wait_for_selector(".js-plotly-plot", state="attached", timeout=20000)
    else:
        page.wait_for_selector('[data-testid="stDataFrame"]', state="attached", timeout=20000)
    page.wait_for_timeout(1000)
    body_text = page.locator("body").inner_text()
    if any(marker in body_text for marker in ("Running...", "Loading...", "Please wait")):
        page.wait_for_timeout(2000)


def main():
    data_manifest = verify_market_data()
    print(f"Starting Streamlit server on port {PORT} with dark theme...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.backgroundColor", "#0b0f17",
        "--theme.secondaryBackgroundColor", "#111722",
        "--theme.textColor", "#f1f5f9"
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        print("Waiting for server ready...")
        if not wait_for_server(f"http://localhost:{PORT}/_stcore/health", timeout=35):
            raise TimeoutError("Streamlit server failed to start in time.")

        print("Connecting Playwright browser...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 980},
                device_scale_factor=1,
            )
            page = context.new_page()
            print(f"Navigating to {URL}...")
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Collapse the sidebar before captures.
            close_sidebar(page)
            page.wait_for_selector(".summary-row", timeout=30000)
            context_text = page.locator(".context-bar").inner_text()
            summary_text = page.locator(".summary-row").inner_text()
            print("Verified baseline run context:\n", context_text)
            print("Verified summary metrics:\n", summary_text)

            # 1. Backtest view (charts)
            print("Capturing charts (Backtest view)...")
            prepare_view(page, "Backtest")
            raw1 = ROOT / "scripts" / "tmp_charts.png"
            page.screenshot(path=str(raw1), clip={"x": 0, "y": 0, "width": 1920, "height": 980})
            save_screenshot_formats(raw1, "charts")

            # 2. Validation view (validation)
            print("Capturing validation (Walk-Forward Validation view)...")
            prepare_view(page, "Validation")
            page.wait_for_selector("text=Stitched OOS return", timeout=20000)
            page.wait_for_timeout(1500)
            raw2 = ROOT / "scripts" / "tmp_validation.png"
            page.screenshot(path=str(raw2), clip={"x": 0, "y": 0, "width": 1920, "height": 980})
            save_screenshot_formats(raw2, "validation")

            # 3. Risk view (charts-more)
            print("Capturing charts-more (Risk view)...")
            prepare_view(page, "Risk", requires_plot=False)
            raw3 = ROOT / "scripts" / "tmp_charts_more.png"
            page.screenshot(path=str(raw3), clip={"x": 0, "y": 0, "width": 1920, "height": 980})
            save_screenshot_formats(raw3, "charts-more")

            # 4. Signals view (signal)
            print("Capturing signal (Signals view)...")
            prepare_view(page, "Signals")
            raw4 = ROOT / "scripts" / "tmp_signal.png"
            page.screenshot(path=str(raw4), clip={"x": 0, "y": 0, "width": 1920, "height": 980})
            save_screenshot_formats(raw4, "signal")

            # 5. Allocations view (weights)
            print("Capturing weights (Allocations view)...")
            prepare_view(page, "Allocations")
            raw5 = ROOT / "scripts" / "tmp_weights.png"
            page.screenshot(path=str(raw5), clip={"x": 0, "y": 0, "width": 1920, "height": 980})
            save_screenshot_formats(raw5, "weights")

            # Clean up temp files
            for tmp in [raw1, raw2, raw3, raw4, raw5]:
                if tmp.exists():
                    tmp.unlink()

            manifest = {
                "requested_period": {"start": "2019-01-01", "end": "2026-09-18"},
                "cached_dataset": data_manifest,
                "portfolio_screenshots": {
                    "charts": BASELINE_STRATEGY,
                    "validation": BASELINE_STRATEGY,
                    "charts-more": BASELINE_STRATEGY,
                    "signal": BASELINE_STRATEGY,
                    "weights": BASELINE_STRATEGY,
                },
                "strategy": BASELINE_STRATEGY,
            }
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Wrote screenshot provenance manifest: {MANIFEST_PATH}")

            browser.close()
            print("All screenshots successfully captured, validated, and converted!")

    finally:
        print("Stopping Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
