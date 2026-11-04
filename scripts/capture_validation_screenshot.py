from __future__ import annotations

import subprocess
import time
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8522
URL = f'http://localhost:{PORT}/?preload_validation=1'


def main():
    print(f'Starting Streamlit process on port {PORT}...')
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        'app.py',
        '--server.port',
        str(PORT),
        '--server.headless',
        'true',
        '--browser.gatherUsageStats',
        'false',
        '--theme.base',
        'dark',
        '--theme.backgroundColor',
        '#0b0f17',
        '--theme.secondaryBackgroundColor',
        '#111722',
        '--theme.textColor',
        '#f1f5f9',
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        import urllib.request
        ready = False
        print('Waiting for Streamlit server health...')
        for _ in range(50):
            try:
                with urllib.request.urlopen(f'http://localhost:{PORT}/_stcore/health', timeout=1) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)

        if not ready:
            print('Streamlit server failed to start in time.')
            return 1

        print('Streamlit server is ready. Launching Playwright browser...')
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 980},
                device_scale_factor=1.0,
            )
            page = context.new_page()
            print(f'Navigating to {URL}...')
            page.goto(URL, wait_until='networkidle')
            page.wait_for_timeout(2000)

            # Collapse the sidebar before capture.
            close_btn = page.locator('button[data-testid="stSidebarCollapseButton"]')
            if close_btn.count() and close_btn.first.is_visible():
                print('Collapsing sidebar...')
                close_btn.first.click()
                page.wait_for_timeout(600)

            # Switch to Validation tab
            print('Switching to Validation tab...')
            val_tab = page.get_by_role('tab', name='Validation', exact=True)
            val_tab.wait_for(state='visible', timeout=20000)
            val_tab.click()
            page.wait_for_timeout(1000)

            # Wait for key validation elements and plots
            print('Waiting for validation metrics and plotly chart...')
            page.wait_for_selector('text=Stitched OOS return', timeout=30000)
            page.wait_for_selector('.js-plotly-plot', state='attached', timeout=20000)
            page.evaluate('window.scrollTo(0, 0)')
            page.wait_for_timeout(2500)

            # Capture screenshot
            out_png = ROOT / 'docs' / 'validation.png'
            out_png.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out_png), clip={'x': 0, 'y': 0, 'width': 1920, 'height': 980})
            print(f'Successfully captured screenshot to: {out_png}')

            # Convert to WebP for portfolio site
            portfolio_webp = Path(r'C:\Projects\Portfolio-site\assets\images\projects\algorithmic-trading-backtester\validation.webp')
            portfolio_webp.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(out_png) as im:
                im.save(portfolio_webp, 'WEBP', quality=92, method=6)
            print(f'Successfully converted and saved webp to: {portfolio_webp}')

            browser.close()
            print('Browser closed. Screenshot workflow completed successfully!')

    finally:
        print('Stopping Streamlit process...')
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    return 0


if __name__ == '__main__':
    sys.exit(main())
