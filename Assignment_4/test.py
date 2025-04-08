import os
import time
import requests
import subprocess

def wait_for_server(url, timeout=30):
    print("⏳ Waiting for server to start...")
    for _ in range(timeout):
        try:
            response = requests.get(url)
            if response.status_code in [200, 404]:  # 404 means root route not defined but server is up
                print("✅ Server is up!")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("❌ Server did not start within the expected time.")

def test_docker():
    print("🔨 Building Docker image...")
    subprocess.run("docker build -t flask-app .", shell=True, check=True)

    print("🚀 Running Docker container...")
    subprocess.run("docker run -d -p 5000:5000 --name test_container flask-app", shell=True, check=True)

    try:
        wait_for_server("http://localhost:5000")

        print("📡 Sending POST request to /score...")
        url = "http://localhost:5000/score"
        sample_input = {"text": "I love Python!"}
        response = requests.post(url, json=sample_input)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "label" in data or "prediction" in data, "Response JSON missing expected key"

        print("✅ Test passed. Response:", data)

    finally:
        print("🧹 Cleaning up Docker container...")
        subprocess.run("docker stop test_container", shell=True)
        subprocess.run("docker rm test_container", shell=True)

if __name__ == "__main__":
    test_docker()
