import sys
from generate import generate

name = sys.argv[1]

generate(
    f"lora/{name}.safetensors",
    f"generated/{name}",
    "1girl, portrait, ultra realistic"
)