from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

def ollamaParserClient(text, template , model='iodose/nuextract-v1.5'):
    response = client.chat(model=model, messages=[
        {
            'role': 'user',
            'content': f'{template}\n\n{text}',
        },
    ])
    return response['message']['content']