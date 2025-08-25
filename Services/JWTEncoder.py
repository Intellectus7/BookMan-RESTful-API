from jose import jwt
from datetime import datetime, timedelta

SECRET = "park-secret"
ALGORITHM = "HS256"

class JWT:
    def __init__(self, secret=SECRET, alg=ALGORITHM, claims=None):
        self.secret = secret
        self.alg = alg
        self.claims = claims or {}
        self.token = None

    def encode(self):
        self.token = jwt.encode(self.claims, self.secret, algorithm=self.alg)
        return self.token

    def decode_value(self, encoded_value=None):
        token_to_decode = encoded_value or self.token
        return jwt.decode(token_to_decode, self.secret, algorithms=[self.alg])

    def verify(self, encoded_value=None):
        try:
            decoded = self.decode_value(encoded_value)
            return True, decoded
        except Exception as e:
            return False, str(e)


# 🎟 Example usage
claims = {
    "sub": "visitor123",
    "role": "VIP",
    "exp": datetime.utcnow() + timedelta(hours=12)
}

helper = JWT(claims=claims)
token = helper.encode()
print("Token:", token)

ok, result = helper.verify(token)
print("Valid?" , ok)
print("Decoded:", result)
