import os
import PyPDF2 as p2


directory = 'C:\\Users\\Millind\\Desktop\\Study-Material\\Modern-CPP-Programming'

pdf_files = []

for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        pdf_files.append(directory + '\\' + filename)

pdf_files.sort(key=str.lower)

dest_file = directory + '\\' + 'Modern-CPP-Programming.pdf'

pdf_writer = p2.PdfFileWriter()

for filename in pdf_files:
    print("Adding: " + filename)
    pdf_reader = p2.PdfFileReader(filename)
    for page in range(pdf_reader.getNumPages()):
        pdf_writer.addPage(pdf_reader.getPage(page))


pdf_writer.write(dest_file)
