import os
import requests
from dotenv import load_dotenv

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def test_sarvam_stt():
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    try:
        # Just a dummy probe to see if we get 401/403 or 400 (Bad Request)
        response = requests.post(url, headers=headers)
        print(f"STT Probe Status: {response.status_code}")
        print(f"STT Probe Response: {response.text}")
    except Exception as e:
        print(f"STT Probe Failed: {e}")

if __name__ == "__main__":
    test_sarvam_stt()
