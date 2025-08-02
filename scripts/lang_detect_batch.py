import sys
import json
from langdetect import detect

def main():
    input_data = json.loads(sys.stdin.read())
    texts = input_data.get('texts', [])
    results = []
    for text in texts:
        try:
            lang = detect(text)
        except Exception:
            lang = 'unknown'
        results.append({'lang': lang})
    print(json.dumps(results))

if __name__ == '__main__':
    main() 