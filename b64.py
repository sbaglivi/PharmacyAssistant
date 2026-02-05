from base64 import b64encode
import sys

text = " ".join(sys.argv[1:])


print(b64encode(text.encode()).decode())