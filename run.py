"""Start the web app:  python run.py   ->  http://127.0.0.1:8000"""
import threading
import webbrowser

import uvicorn

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Image Noise Removal Lab running at {url}  (Ctrl+C to stop)")
    uvicorn.run("server:app", host="127.0.0.1", port=8000)
