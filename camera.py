import requests
import os
from datetime import datetime
import time
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def capture_snapshot(camera_url, output_dir="snapshots", username=None, password=None, timeout=5):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'image/jpeg,image/png,image/*,*/*',
            'Connection': 'close'
        }

        response = requests.get(
            camera_url,
            auth=HTTPDigestAuth(username, password) if username and password else None,
            timeout=timeout,
            headers=headers,
            verify=False,
            stream=True
        )

        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                print(f"[CAMERA ERROR] Unexpected content-type: {content_type}")
                print(f"[CAMERA DEBUG] Response: {response.text[:200]}...")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"speeding_{timestamp}.jpg")

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        f.write(chunk)

            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                if file_size < 1024:
                    os.remove(filename)
                    print(f"[CAMERA ERROR] Snapshot too small: {file_size} bytes")
                    return None
                return filename
        return None

    except Exception:
        return None
