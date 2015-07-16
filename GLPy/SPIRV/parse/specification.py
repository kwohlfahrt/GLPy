from lxml import etree
from pathlib import Path

SPIRV_URL = "https://www.khronos.org/registry/spir-v/specs/1.0/SPIRV.html"
SPIRV_FILE = Path(__file__).with_name("SPIRV.html")

try:
    with SPIRV_FILE.open('r'):
        pass
except FileNotFoundError:
    from requests import get

    r = get(SPIRV_URL)
    r.raise_for_status()
    with SPIRV_FILE.open('wb') as f:
        for chunk in r:
            f.write(chunk)

spec_tree = etree.parse(str(SPIRV_FILE), etree.HTMLParser())

def normalizeWhitespace(*strings):
    return ' '.join(''.join(strings).split())

def text(element):
    return normalizeWhitespace(etree.tostring(element, method='text').decode())
