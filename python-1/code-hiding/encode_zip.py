from zipfile import ZipFile
from cryptography.fernet import Fernet

# Load the key from the current directory
with open("keys/key.bin", "rb") as key_file:
    key = key_file.read()

# read code/User.java
with open("code/User.java", "rb") as user_file:
    user = user_file.read()

# zip code/User.java with password 123
with ZipFile("code/User.zip", "w") as user_zip_file:
    user_zip_file.setpassword(b"123")
    user_zip_file.write("code/User.java")
