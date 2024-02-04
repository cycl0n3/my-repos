import os

def add_pdf_extension(directory):
  for filename in os.listdir(directory):
    if not os.path.isfile(os.path.join(directory, filename)):
      continue
    name, ext = os.path.splitext(filename)
    if not ext:
      new_filename = name + ".pdf"
      os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
      #print(new_filename)

# Example usage
directory = "C:\\Users\\Millind\\Downloads\\71000"
add_pdf_extension(directory)
