from cryptography.fernet import Fernet

# Load the key from the current directory
with open("keys/key.bin", "rb") as key_file:
    key = key_file.read()

# read code/User.enc.java
with open("code/User.enc.java", "rb") as user_enc_file:
    user_enc = user_enc_file.read()


# decrypt code/User.enc.java
f = Fernet(key)
decrypted = f.decrypt(user_enc)

# write decrypted code/User.dec.java
with open("code/User.dec.java", "wb") as user_dec_file:
    user_dec_file.write(decrypted)
