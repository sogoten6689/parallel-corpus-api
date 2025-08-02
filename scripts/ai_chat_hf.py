import sys
import json
import requests

def main():
    input_data = json.loads(sys.stdin.read())
    question = input_data['question']
    data = input_data.get('data', None)
    chat_history = input_data.get('chatHistory', [])

    # Build context from chat history (last 5 messages)
    context = ""
    if chat_history:
        recent_messages = chat_history[-5:]
        for msg in recent_messages:
            role = "User" if msg.get('type') == 'user' else "Assistant"
            content = msg.get('content', '')[:150]  # Limit length for speed
            context += f"{role}: {content}\n"

    # Create optimized prompt for faster response
    if data and isinstance(data, dict) and 'fileName' in data and 'preview' in data:
        # Simplified file info for speed
        file_info = f"File: {data.get('fileName', '')}, {data.get('rows', '')} rows, {data.get('columns', '')} cols"
        if data.get('columnNames'):
            file_info += f", Columns: {', '.join(data.get('columnNames', [])[:5])}"  # Only first 5 columns
        
        prompt = f"Data: {file_info}\nHistory: {context[:200]}\nQ: {question}\nAnswer briefly:"
    elif data:
        prompt = f"Data: {str(data)[:200]}\nHistory: {context[:200]}\nQ: {question}\nAnswer:"
    else:
        prompt = f"History: {context[:300]}\nQ: {question}\nAnswer:"

    # Gọi Ollama API với model 'mistral' và tối ưu cho tốc độ
    API_URL = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,  # Limit response length for speed
            "top_k": 20,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        answer = result.get('response', '').strip()
        if not answer:
            answer = "Xin lỗi, tôi không thể trả lời câu hỏi này."
    except Exception as e:
        print("DEBUG: Ollama response error:", str(e), file=sys.stderr)
        answer = "Xin lỗi, tôi không thể trả lời câu hỏi này."

    print(json.dumps({"answer": answer}))

if __name__ == '__main__':
    main() 