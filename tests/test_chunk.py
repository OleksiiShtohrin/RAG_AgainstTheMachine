import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.chunking import Chunk

chunk = Chunk(
    file_path="data/raw/example.py",
    content="def hello():\n    pass",
    first_character_index=0,
    last_character_index=21,
)

print("Test 1 - OK:\n")
print(chunk)


print("Test 2 - KO(ValueError):\n")
chunk = Chunk(
    file_path="data/raw/example.py",
    content="hello",
    first_character_index=-1,
    last_character_index=5,
)
print(chunk)


print("Test 3 - KO(ValueError):\n")
chunk = Chunk(
    file_path="data/raw/example.py",
    content="hello",
    first_character_index=10,
    last_character_index=5,
)
print(chunk)
