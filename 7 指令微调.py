import json
import os
import urllib
import urllib.request
from pydantic import Json


def download_and_load_file(file_path: str, url: str) -> Json:
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data: str = file.read()
    with open(file_path, 'r') as file:
        data: Json = json.load(file)
    return data
