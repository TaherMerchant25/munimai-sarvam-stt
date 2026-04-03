import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

def test_endpoint(url, filename):
    files = {'file': open(filename, 'rb')}
    print(f"\nTesting {url} with {filename}...")
    try:
        response = requests.post(url, files=files)
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error:", e)

for file in ["238.mp3", "262.mp3"]:
    test_endpoint("http://localhost:8001/api/audio/transcript", file)
    test_endpoint("http://localhost:8001/api/audio/process", file)
