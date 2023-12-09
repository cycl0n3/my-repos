from cryptography.fernet import Fernet

# Load the key from the current directory
with open("keys/key.bin", "rb") as key_file:
    key = key_file.read()

# read code/User.java
with open("code/User.java", "rb") as user_file:
    user = user_file.read()

# encrypt code/User.java
f = Fernet(key)
encrypted = f.encrypt(user)

# write encrypted code/User.enc.java
with open("code/User.enc.java", "wb") as user_enc_file:
    user_enc_file.write(encrypted)
