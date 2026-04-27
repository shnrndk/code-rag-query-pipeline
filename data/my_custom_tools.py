# data/my_custom_tools.py
import json
import csv
import subprocess
from bs4 import BeautifulSoup
from openai import OpenAI

def scrape_html_text(html_content):
    """
    Extract and scrape text data from an HTML web page using BeautifulSoup.
    This strips all tags and returns clean text.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def generate_automated_unit_test(function_code, api_key):
    """
    Generate an automated unit test for a python function using an LLM.
    Uses the OpenAI API to write pytest compatible test cases.
    """
    client = OpenAI(api_key=api_key)
    prompt = f"Write a pytest unit test for the following Python function:\n\n{function_code}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def parse_json_config(file_path):
    """
    Read and parse a JSON configuration file.
    Returns the parsed dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return config_data

def write_dict_to_csv(data_list, output_file):
    """
    Write a list of dictionary data out to a CSV file.
    Extracts headers automatically from the first dictionary's keys.
    """
    if not data_list:
        return
    headers = data_list[0].keys()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_list)

def execute_shell_with_timeout(command, timeout_seconds=30):
    """
    Execute a shell command as a subprocess with a strict timeout limit.
    Returns the standard output as a string.
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout_seconds
    )
    return result.stdout