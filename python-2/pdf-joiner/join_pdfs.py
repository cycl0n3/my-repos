import os
from PyPDF2 import PdfFileMerger

def join_pdfs(directory):
  merger = PdfFileMerger()
  pdf_files = sorted([file for file in os.listdir(directory) if file.endswith('.pdf')])

  for file in pdf_files:
    file_path = os.path.join(directory, file)
    print(f'Adding {file_path}...')
    merger.append(file_path)

  output_path = os.path.join(directory, 'merged.pdf')
  merger.write(output_path)
  merger.close()

  print(f'Merged PDF saved to {output_path}')

# Usage example
directory = 'C:\\Users\\Millind\\Books\\ART'
join_pdfs(directory)
