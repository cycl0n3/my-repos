from bs4 import BeautifulSoup

import requests
import os

import tqdm

from urllib.parse import unquote

folder_name = 'Electronics'

# Create a folder to store the pdf files
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# Get the url of the website
url = 'https://theswissbay.ch/pdf/Gentoomen%20Library/Electronics/'

# Get the html content of the website
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, 'lxml')

# Get all the links in the website
links = soup.find_all('a')

# Create a list of extensions
extensions = ['.pdf', '.zip', '.chm', '.epub', '.djvu', '.mobi', '.rar']

# Iterate through all the links
for link in tqdm.tqdm(links, colour='green', desc='Downloading'):
    # Get the link href text and decode it
    link_text = unquote(link['href'])

    # Check if the link text ends with any of the extensions
    if link_text.endswith(tuple(extensions)):
        # Get the link url
        link_url = url + link['href']

        # Check if the file already exists
        if os.path.exists(f'{folder_name}/{link_text}'):
            continue

        # Temporary file name
        unconfirmed_file = f'{folder_name}/{link_text}.unconfirmed'

        # Actual file name
        confirmed_file = f'{folder_name}/{link_text}'

        # Write contents to file with random name
        with open(unconfirmed_file, 'wb') as f:
            # Get the pdf content in chunks
            content = requests.get(link_url, stream=True).content
            f.write(content)

        # Read the file content and write to actual file
        with open(unconfirmed_file, 'rb') as f:
            content = f.read()

            # Create a file with the link text
            with open(confirmed_file, 'wb') as f:
                # Write the pdf content to the file
                f.write(content)
        
        # Delete the file with random name
        os.remove(unconfirmed_file)

