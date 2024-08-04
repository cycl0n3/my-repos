import requests
from bs4 import BeautifulSoup
import os
import random
import string

def download_pdfs(url):
    try:
        # Create a random folder name
        folder_name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        os.mkdir(folder_name)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        print(response)

        # Find all <a> tags with href ending in .pdf
        pdf_links = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))

        for link in pdf_links:
            pdf_url = link['href']
            try:
                pdf_response = requests.get(pdf_url, stream=True)
                pdf_filename = pdf_url.split('/')[-1]
                file_path = os.path.join(folder_name, pdf_filename)

                with open(file_path, 'wb') as pdf_file:
                    for chunk in pdf_response.iter_content(chunk_size=1024):
                        if chunk:
                            pdf_file.write(chunk)
                print(f"Downloaded: {pdf_filename}")
            except Exception as e:
                print(f"Error downloading {pdf_url}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'https://example.com' with the actual URL
url = 'http://sanskritdocuments.org/sanskrit/vishhnu/'
download_pdfs(url)
