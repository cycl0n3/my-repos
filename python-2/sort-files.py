import os
import unicodedata

dir = 'C:\\Users\\Millind\\Books\\MN_Numerical methods'

# Get the list of all files in directory tree at given path
listOfFiles = list()

for (dirpath, dirnames, filenames) in os.walk(dir):
  listOfFiles += [os.path.join(dirpath, file) for file in filenames]

englishFiles = []
russianFiles = []

for elem in listOfFiles:
  class_name = 'english'

  for char in elem:
    if 'Cyrillic' in unicodedata.name(char, ''):
      class_name = 'russian'
      break
  
  if class_name == 'english':
    englishFiles.append(elem)
  else:
    russianFiles.append(elem)

print("Total English Files", len(englishFiles))
print("Total Russian Files", len(russianFiles))
