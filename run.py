import requests
import os
import json
import subprocess
import time

from config import  *


def main():
    """Process all content in wait list."""
    # get all content wait list
    content_list = requests.get(f"{ROOT_API}/content_command/get_wait")
    content_list = content_list.json()
    content_list = sorted(
        content_list,
        key=lambda x: x["command"],
        reverse=True
    )
    for content in content_list:
        content_list_json = json.dumps(
            content,
            indent=4,
            ensure_ascii=False
        )
        with open("content_info.json", "w") as f:
            f.write(content_list_json)
        subprocess.call("python3 analysis_process.py", shell=True)
    
    # remove all file images, audios, videos
    print("REMOVE FILES: audios/*, images/*, videos/*, subs/*")
    subprocess.call("rm -rf audios/*", shell=True)
    subprocess.call("rm -rf images/*", shell=True)
    subprocess.call("rm -rf videos/*", shell=True)
    subprocess.call("rm -rf subs/*", shell=True)
    subprocess.call("rm -rf features/img_features/*", shell=True)
    subprocess.call("rm -rf features/audio_features/*", shell=True)
    
    # Wait for new content
    print("Sleeping for one minutes...")
    time.sleep(30)
        
if __name__ == "__main__":
    while True:
        main()