"""
Screenshot Capture Module
Captures high-quality screenshots of Power BI dashboards using Selenium with Edge.
Also supports manual screenshot upload as a fallback.
"""

import os
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


# Optional Streamlit app URL to allow capturing the Streamlit page itself (default localhost)
STREAMLIT_URL = os.getenv("STREAMLIT_URL") or "http://localhost:8501"
SCREENSHOT_DIR = "screenshots"

# Simple local HTTP receiver to accept base64 image POSTs from the injected JS
_receiver_thread = None
_receiver_port = 8765


class _UploadHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        if self.path != '/upload':
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'not found'}).encode('utf-8'))
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
            token = payload.get('token') or datetime.now().strftime('%Y%m%d_%H%M%S')
            img_data = payload.get('image')
            if not img_data:
                raise ValueError('No image data')
            # Strip header
            prefix = 'data:image/png;base64,'
            if img_data.startswith(prefix):
                b64 = img_data[len(prefix):]
            else:
                b64 = img_data
            import base64
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            filename = f"internal_{token}.png"
            path = os.path.join(SCREENSHOT_DIR, filename)
            with open(path, 'wb') as f:
                f.write(base64.b64decode(b64))
            self._set_headers(200)
            self.wfile.write(json.dumps({'status': 'ok', 'path': path}).encode('utf-8'))
            print(f"✅ Received and saved internal screenshot: {path}")
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            print('❌ Upload handler error:', e)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def ensure_receiver_running(port: int = None):
    global _receiver_thread, _receiver_port
    if port:
        _receiver_port = port
    if _receiver_thread and _receiver_thread.is_alive():
        return _receiver_port

    def _run():
        server_address = ('127.0.0.1', _receiver_port)
        httpd = ThreadedHTTPServer(server_address, _UploadHandler)
        print(f"📡 Internal receiver listening on http://{server_address[0]}:{server_address[1]}/upload")
        try:
            httpd.serve_forever()
        except Exception:
            pass

    _receiver_thread = threading.Thread(target=_run, daemon=True)
    _receiver_thread.start()
    # give it a short moment to start
    time.sleep(0.1)
    return _receiver_port


def detect_streamlit_url():
    """Try to detect the correct Streamlit URL by testing common ports."""
    try:
        import requests
    except Exception:
        # requests not available; fall back to default
        return STREAMLIT_URL

    # Common Streamlit ports to try
    ports_to_try = [8501, 8502, 8503, 8000, 3000]

    for port in ports_to_try:
        test_url = f"http://localhost:{port}"
        try:
            response = requests.get(test_url, timeout=2)
            # Accept 200 or redirects; Streamlit page sometimes doesn't include 'streamlit' in body
            if response.status_code in (200, 302):
                print(f"✅ Found web server at {test_url} (status {response.status_code})")
                return test_url
        except requests.exceptions.RequestException:
            continue
    
    # Fallback to environment variable or default
    return STREAMLIT_URL


def capture_screenshot_internal(target_selector: str = "#powerbi-report-capture", key: str = "screenshot_capture") -> str:
    """
    Capture screenshot using JavaScript html2canvas from inside Streamlit.
    This is FAST and requires no external browser.
    
    Args:
        target_selector: CSS selector for element to capture (default: 'iframe' for Power BI)
        key: Unique key for the component (to maintain state across reruns)
        
    Returns:
        Base64 image data string (data:image/png;base64,...) or None if capture failed
    """
    import streamlit.components.v1 as components

    # Ensure the local HTTP receiver is running to accept the posted image
    port = ensure_receiver_running()
    token = datetime.now().strftime('%Y%m%d_%H%M%S')
    receiver_url = f'http://127.0.0.1:{port}/upload'

    # HTML/JS component that loads html2canvas and POSTs the base64 image JSON to our receiver
    capture_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    </head>
    <body>
        <div id="status" style="padding: 10px; font-family: sans-serif; font-size: 14px;">Initializing capture...</div>
        <script>
            async function captureScreenshot() {{
                try {{
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = '⏳ Finding target element...';

                    const targetSelector = '{target_selector}';
                    let targetElement = window.parent.document.querySelector(targetSelector);
                    if (!targetElement) {{
                        statusEl.textContent = '⚠️ Target not found, capturing full page...';
                        targetElement = window.parent.document.body;
                    }}

                    statusEl.textContent = '📸 Capturing report container...';
                    // Use lightweight options for speed: lower scale, avoid foreignObjectRendering
                    const canvas = await html2canvas(targetElement, {{
                        scale: 1,
                        backgroundColor: null,
                        logging: false,
                        foreignObjectRendering: false
                    }});
                    const imgData = canvas.toDataURL('image/png');

                    statusEl.textContent = '📤 Sending to local receiver...';
                    const payload = {{ token: '{token}', image: imgData }};
                    await fetch('{receiver_url}', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
                    statusEl.textContent = '✅ Capture sent';
                }} catch (err) {{
                    document.getElementById('status').textContent = '❌ ' + err.message;
                    console.error('Capture error:', err);
                }}
            }}
            // Run the capture almost immediately; user is responsible for clicking when report is ready
            setTimeout(captureScreenshot, 200);
        </script>
    </body>
    </html>
    """

    # Render the component (fires JS asynchronously). Caller should poll for the saved file.
    try:
        # Keep the injected component very small to avoid layout impact
        components.html(capture_html, height=40)
    except Exception as e:
        print('❌ components.html error:', e)
        return None

    expected_path = os.path.join(SCREENSHOT_DIR, f"internal_{token}.png")
    return expected_path


def save_base64_screenshot(base64_data: str) -> str:
    """
    Save a base64-encoded screenshot to file.
    
    Args:
        base64_data: Base64 image data string (data:image/png;base64,...)
        
    Returns:
        Path to saved screenshot file
    """
    import base64
    import re
    
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"internal_{timestamp}.png")
    
    # Remove data URL prefix if present
    base64_pattern = r'^data:image/\w+;base64,'
    base64_clean = re.sub(base64_pattern, '', base64_data)
    
    # Decode and save
    image_bytes = base64.b64decode(base64_clean)
    with open(screenshot_path, 'wb') as f:
        f.write(image_bytes)
    
    print(f"✅ Saved internal screenshot to: {screenshot_path}")
    return screenshot_path


def save_uploaded_screenshot(uploaded_file) -> str:
    """
    Save an uploaded screenshot file to the screenshots directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object with .read() and .name
        
    Returns:
        Path to the saved screenshot
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get file extension from uploaded file name
    original_name = getattr(uploaded_file, 'name', 'screenshot.png')
    ext = os.path.splitext(original_name)[1] or '.png'
    
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"uploaded_{timestamp}{ext}")
    
    # Save the file
    with open(screenshot_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    print(f"✅ Saved uploaded screenshot to: {screenshot_path}")
    return screenshot_path


def _setup_selenium():
    """Lazy import and setup for Selenium (only when needed for automated capture)."""
    from selenium import webdriver
    from selenium.webdriver.edge.service import Service
    from selenium.webdriver.edge.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    return webdriver, Service, Options, By, WebDriverWait, EC, EdgeChromiumDriverManager


def capture_screenshot_sync(wait_time: int = 20, debug: bool = False, proxy: str | None = None, user_data_dir: str | None = None, container_selector: str | None = None, streamlit_capture: bool = False, streamlit_url: str | None = None) -> str:
    """
    Captures a high-quality screenshot of the Power BI dashboard.
    Uses Selenium with Microsoft Edge for reliable Windows compatibility.
    
    Args:
        wait_time: Seconds to wait for dashboard to fully load
        
    Returns:
        Path to the saved screenshot
    """
    # Lazy import Selenium modules
    webdriver, Service, Options, By, WebDriverWait, EC, EdgeChromiumDriverManager = _setup_selenium()
    
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"dashboard_{timestamp}.png")
    
    # Configure Edge options
    edge_options = Options()
    # If debug is True, run with visible browser to help troubleshooting
    if debug:
        # non-headless for debugging
        edge_options.add_argument("--headless=disabled")
    else:
        edge_options.add_argument("--headless")  # New headless mode
    edge_options.add_argument("--window-size=1920,1080")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0")
    
    # For higher quality screenshots
    edge_options.add_argument("--force-device-scale-factor=2")
    # If proxy not provided explicitly, try environment variables
    if proxy is None:
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    if proxy:
        print(f"🔐 Using proxy: {proxy}")
        try:
            edge_options.add_argument(f"--proxy-server={proxy}")
        except Exception as e:
            print("⚠️ Failed to set proxy option:", e)
    # If a user data dir is provided, instruct Edge to use it (keeps your logged-in session)
    if user_data_dir:
        user_data_dir = os.path.expanduser(user_data_dir)
        print(f"👤 Using Edge user data dir: {user_data_dir}")
        try:
            edge_options.add_argument(f"--user-data-dir={user_data_dir}")
        except Exception as e:
            print("⚠️ Failed to set user_data_dir option:", e)
    
    driver = None
    try:
        print(f"🌐 Launching Microsoft Edge browser...")

        # Use webdriver-manager to automatically handle EdgeDriver
        service = Service(EdgeChromiumDriverManager().install())
        driver = webdriver.Edge(service=service, options=edge_options)

        # Set window size for consistent screenshots
        driver.set_window_size(1920, 1080)

        # Decide whether to capture the Power BI embed URL directly, or the Streamlit page
        if streamlit_capture:
            if streamlit_url:
                target = streamlit_url
            else:
                # Try to auto-detect the correct Streamlit URL
                target = detect_streamlit_url()
            print(f"🌐 Navigating to Streamlit app: {target}")
        
        try:
            driver.get(target)
        except Exception as nav_error:
            if streamlit_capture:
                # If auto-detection failed, try the manual fallback URLs
                fallback_urls = ["http://localhost:8501", "http://127.0.0.1:8501"]
                success = False
                for fallback_url in fallback_urls:
                    try:
                        print(f"🔄 Trying fallback URL: {fallback_url}")
                        driver.get(fallback_url)
                        success = True
                        break
                    except Exception:
                        continue
                
                if not success:
                    raise Exception(f"Could not connect to Streamlit app. Tried: {target}, {', '.join(fallback_urls)}. Make sure Streamlit is running and accessible.")
            else:
                raise nav_error

        if streamlit_capture:
            print("🚀 Streamlit capture mode - page already loaded, capturing immediately")
            # For Streamlit captures, just wait a moment for page to settle
            time.sleep(0.5)
        else:
            print(f"⏳ Waiting for Power BI dashboard to load and render...")
            # For Power BI direct captures, use the original wait logic
            time.sleep(1)
        
        # Wait intelligently for the content to be ready (only for Power BI direct captures)
        if not streamlit_capture:
            try:
                if container_selector:
                    WebDriverWait(driver, min(15, wait_time)).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, container_selector))
                    )
                else:
                    # Fallback: wait for canvas or iframe elements which indicate visuals are present
                    WebDriverWait(driver, min(15, wait_time)).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "iframe, canvas, .visualContainer, .visual"))
                    )
                print("✅ Visual container detected")
            except Exception as wait_exc:
                print("⚠️ Could not detect specific visuals immediately, continuing...", str(wait_exc))

            # Then poll for canvas content or page readiness up to the full wait_time
            end_time = time.time() + wait_time
            while time.time() < end_time:
                try:
                    ready = driver.execute_script("""
                    if (document.readyState !== 'complete') return false;
                    const canvases = Array.from(document.querySelectorAll('canvas'));
                    if (!canvases.length) return true;
                    return canvases.some(c => { try { return c.toDataURL().length > 1000 } catch(e) { return false } });
                    """)
                    if ready:
                        break
                except Exception:
                    pass
                time.sleep(0.5)
        
        # If a specific container selector is provided, try to capture that element
        if container_selector:
            try:
                print(f"🔎 Looking for container selector: {container_selector}")
                # Wait up to 15s for the element
                elem = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, container_selector))
                )
                # Use WebElement.screenshot to capture just the element (works for iframe element visuals)
                elem.screenshot(screenshot_path)
                print(f"📸 Element screenshot saved: {screenshot_path}")
            except Exception as elem_exc:
                print(f"⚠️ Could not capture element '{container_selector}': {elem_exc}. Falling back to full-page screenshot.")
                driver.save_screenshot(screenshot_path)
        else:
            # Take full-page screenshot
            driver.save_screenshot(screenshot_path)
        print(f"📸 Screenshot saved: {screenshot_path}")
        # Save page HTML for debugging
        try:
            html_path = os.path.join(SCREENSHOT_DIR, f"dashboard_{timestamp}.html")
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(driver.page_source)
            print(f"📝 Page HTML saved: {html_path}")
        except Exception as html_exc:
            print("⚠️ Failed to save page HTML:", html_exc)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Try to capture page source and a partial screenshot for debugging
        if driver:
            try:
                # Attempt to save HTML
                html_path = os.path.join(SCREENSHOT_DIR, f"dashboard_{timestamp}.html")
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(driver.page_source)
                print(f"📝 Partial page HTML saved: {html_path}")
            except Exception as html_exc:
                print("⚠️ Could not save partial HTML:", html_exc)
            try:
                driver.save_screenshot(screenshot_path)
                print(f"📸 Partial screenshot saved: {screenshot_path}")
            except Exception as ss_exc:
                print("⚠️ Could not save partial screenshot:", ss_exc)
        else:
            raise e
    finally:
        if driver:
            driver.quit()
    
    return screenshot_path


def capture_screenshot_local(crop: dict | None = None) -> str:
    """Capture the local OS screen (primary monitor) and save to screenshots dir.

    Args:
        crop: Optional dict with keys left, top, width, height to crop the captured image.
    Returns:
        Path to saved screenshot
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"local_{timestamp}.png")

    try:
        # Prefer mss for fast screen capture (use dynamic import to avoid static unresolved-import warnings)
        try:
            import importlib
            mss_lib = importlib.import_module("mss")
            # Import PIL.Image dynamically to avoid static analysis unresolved-import warnings
            pil_image = importlib.import_module("PIL.Image")
            with mss_lib.mss() as sct:
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                img = pil_image.frombytes("RGB", sct_img.size, sct_img.rgb)
        except Exception:
            # Fallback to Pillow's ImageGrab (works on Windows)
            try:
                import importlib as _importlib
                pil_imagegrab = _importlib.import_module("PIL.ImageGrab")
                img = pil_imagegrab.grab()
            except Exception:
                # If Pillow is not available at all, propagate the error to outer handler
                raise

        if crop:
            left = int(crop.get('left', 0))
            top = int(crop.get('top', 0))
            width = int(crop.get('width', img.width))
            height = int(crop.get('height', img.height))
            img = img.crop((left, top, left + width, top + height))

        img.save(screenshot_path)
        print(f"📸 Local screenshot saved: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        print(f"❌ Local capture error: {e}")
        raise


def get_latest_screenshot() -> str | None:
    """Get the most recent screenshot file."""
    if not os.path.exists(SCREENSHOT_DIR):
        return None
    
    screenshots = [
        os.path.join(SCREENSHOT_DIR, f) 
        for f in os.listdir(SCREENSHOT_DIR) 
        if f.endswith('.png')
    ]
    
    if not screenshots:
        return None
    
    return max(screenshots, key=os.path.getctime)


if __name__ == "__main__":
    print("Testing screenshot capture...")
    path = capture_screenshot_sync(wait_time=15)
    print(f"Screenshot captured at: {path}")
