from werkzeug.security import generate_password_hash

hashed_password = generate_password_hash("223")
print(hashed_password)
