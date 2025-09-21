from bs4 import BeautifulSoup

import requests
import os

import tqdm

from urllib.parse import unquote

import time

import re

folder_name = '.'

# Create a folder to store the pdf files
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# Get the url of the website
url = 'https://nzdr.ru/biblio/kolxoz/p/pm'

# Get the html content of the website
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, 'lxml')

# Get all the links in the website
links = soup.find_all('a')

# Create a list of extensions
extensions = ['.pdf', '.zip', '.chm', '.epub', '.djvu', '.mobi', '.rar']

pbar = tqdm.tqdm(links, colour='green', desc='Downloading')

# Iterate through all the links
for link in pbar:
    # Get the link href text and decode it
    link_text = unquote(link['href'])

    pbar.update(1)

    # Check if the link text ends with any of the extensions
    if link_text.endswith(tuple(extensions)) and not "(ru)" in link_text:
        # Get the link url
        link_url = "https://nzdr.ru" + link['href']

        # print something
        # pbar.set_postfix({"Processing": link_url})

        # Constructs the file name
        filename = link_text.split("/")[-1]

        # Remove non-alphanumeric characters and spaces
        filename = re.sub(r'[^\w\s]', '', filename)

        # Replace spaces with dashes
        filename = re.sub(r'\s+', '-', filename)

        # Convert to lowercase
        filename = filename.lower()

        new_filename = filename

        # Replace "_pdf" or "pdf" with ".pdf"
        if filename.endswith('_pdf'):
            new_filename = filename[:-4] + '.pdf'
        elif filename.endswith('pdf'):
            new_filename = filename[:-3] + '.pdf'
        
        # Replace "_djvu" or "djvu" with ".djvu"
        if filename.endswith('_djvu'):
            new_filename = filename[:-5] + '.djvu'
        elif filename.endswith('djvu'):
            new_filename = filename[:-4] + '.djvu'

        filename = new_filename

        # Check if the file already exists
        if os.path.exists(f'{folder_name}/{filename}'):
            continue

        # Temporary file name
        unconfirmed_file = f'{folder_name}/{filename}.unconfirmed'

        # Actual file name
        confirmed_file = f'{folder_name}/{filename}'

        try:
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
        except Exception as e:
            pbar.set_postfix({"Error": e})
        
        # Sleep for 30 seconds
        time.sleep(30)

pbar.close()