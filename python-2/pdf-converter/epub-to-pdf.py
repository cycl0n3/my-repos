import os
import ebooklib
import unicodedata

from ebooklib import epub
from fpdf import FPDF

import unicodedata

def handle_special_characters(text):
    return ''.join(char if unicodedata.category(char)[0] != 'C' else ' ' for char in text)

# Specify the directory path where the EPUB files are located
directory = 'C:\\Users\\Millind\\Books\\Very-New'

# Iterate over all files in the directory
for filename in os.listdir(directory):
  if filename.endswith('.epub'):
    # Print the EPUB file name
    print(f'Converting {filename}...')

    # Open the EPUB file
    book = epub.read_epub(os.path.join(directory, filename))

    # Create a new PDF file
    pdf = FPDF()

    # Iterate over all the chapters in the EPUB file
    for item in book.get_items():
      if item.get_type() == ebooklib.ITEM_DOCUMENT:
        # Extract the chapter content
        content = item.get_content()

        # Convert content to string and ignore encoding errors
        try:
            content_str = content.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error decoding content: {e}")
            content_str = ""

        # Add the chapter content to the PDF file
        pdf.add_page()
        pdf.set_font('Arial', size=12)
        pdf.multi_cell(0, 10, content_str)

    # Save the PDF file with the same name as the EPUB file in binary mode
    with open(os.path.join(directory, filename.replace('.epub', '.pdf')), 'wb') as f:
      pdf.output(f.name, 'F')
