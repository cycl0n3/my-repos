import os
import PyPDF2 as P2

dir = "C:\\Users\\Millind\\Desktop\\Study-Material\\Science2\\The Technician's Radio Receiver Handbook 978-0-7506-7319-8"

# list of pdf files
pdf_files = []

for filename in os.listdir(dir):
    if filename.endswith(".pdf"):
        pdf_files.append(dir + '\\' + filename)

pdf_files.sort(key=str.lower)

print(pdf_files)

dest_file = dir + '\\' + 'The Technician\'s Radio Receiver Handbook 978-0-7506-7319-8.pdf'

pdf_writer = P2.PdfFileWriter()

for filename in pdf_files:
    print("Adding: " + filename)
    pdf_reader = P2.PdfFileReader(filename)
    for page in range(pdf_reader.getNumPages()):
        pdf_writer.addPage(pdf_reader.getPage(page))

pdf_writer.write(dest_file)

print("Done")
