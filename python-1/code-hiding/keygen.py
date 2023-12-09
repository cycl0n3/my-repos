from cryptography.fernet import Fernet

# Generate a key and save it into a file
key = Fernet.generate_key()
with open("keys/key.bin", "wb") as key_file:
    key_file.write(key)
