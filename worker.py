from conductor.client.worker.worker_task import worker_task
from utils.pii import *
from pathlib import Path
from utils.mistralocrr import *
import json
import ast
from utils.groqApplications import *
from utils.indic import *
from utils.ollamaprocesser import ollamaParserClient



@worker_task(task_definition_name='myTask')
def worker(name: str) -> str:
    print(f'Worker called with name: {name}')
    return f'hello, {name}'


@worker_task(task_definition_name='OCRTask')
def ocr_worker(URL: str , TYPE: str) -> str:
    print(f'OCR Worker called with URL: {URL} and TYPE: {TYPE}')
    try:
       return ast.literal_eval(ocr_docu(URL, TYPE))
       
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

@worker_task(task_definition_name='StructuredOCRTask')
def structured_ocr_worker(URL: str) -> str:
    try:
        # return structured_ocr(URL)
        return ast.literal_eval(structured_ocr(URL))
    except Exception as e:
        print(f"Error in Structured OCR processing: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

@worker_task(task_definition_name='transcribeTask')
def transcribe_worker(url: str, model: str = "whisper-large-v3-turbo", prompt: str = None, response_format: str = "verbose_json", timestamp_granularities: list = None, language: str = None, temperature: float = 0.0) :
    try:
        result = transcribe_audio_from_url_groq(
            url=url,
            model=model,
            response_format=response_format,
            timestamp_granularities=["word", "segment"],
            language=language,
            temperature=temperature
        )
        
        # Pretty print the result
        print(result)
        return ast.literal_eval(result)
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"

@worker_task(task_definition_name='piiTask')
def pii_worker(text: str) -> str:
    if text:
        resultjson = {}
        print(f'PII Worker called with text: {text}')
        # sample_text = "John Smith, from London, teaches mathematics at Royal Academy located at 25 King's Road. His employee ID is UK-987654-321 and he has been working there since 2015."
        # sample2 = 'pradeep odela from hyderabad, teaches mathematics at Royal Academy located at 25 King\'s Road. His employee ID is UK-987654-321 and he has been working there since 2015. his credit card number is 1234-5678-9012-3456 and his passport number is A1234567.'
        # Example 1: Using default labels
        results = extract_pii(text)
        print("Example 1: Using default labels")
        print(f"Text: {results['text']}")
        print("Detected entities:")
        for entity in results["entities"]:
            print(f"  {entity['word']} => {entity['entity']} (positions {entity['start']}-{entity['end']})")
        resultjson["text"] = results['text']
        resultjson["entities"] = []
        for entity in results["entities"]:
            entity_dict = {}
            entity_dict["word"] = entity['word']
            entity_dict["entity"] = entity['entity']
            entity_dict["start"] = entity['start']
            entity_dict["end"] = entity['end']
            entity_dict["score"] = entity.get("score", 0)
            resultjson["entities"].append(entity_dict)
        resultss = {
            "text": resultjson["text"],
            "entities": [
                {
                    "entity": entity["entity"],
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity.get("score", 0),
                }
                for entity in resultjson["entities"]
            ],
        }
        # print(resultss)
        return resultss
    else:
        print("No text provided for PII extraction.")
        return "No text provided"





# Task 2: General query processing
@worker_task(task_definition_name='queryTask')
def query_worker(query: str) -> str:

    response = LLMChat(query)
    print(f'Query Worker called with query: {query}')
    print(f'Chat completion response: {response}')
    return response

@worker_task(task_definition_name='InidcToEnglish')
def inidc_worker(text:str,src:str,dst:str) -> str:
    print(f'Indic to English Worker called with text: ')
    print(text)
    print(src)
    print(dst)
    # print(text, src, dst)
    if type(text) is not list:
        text = [text]
    resp = TransulationWorkerIndictoEnglish(text, src_lang=src, tgt_lang=dst)
    print(resp)
    return resp

@worker_task(task_definition_name='StructurdTexttoJson')
def structured_text_to_json_worker(text: str, template: str) -> str:
    try:
        return ast.literal_eval(ollamaParserClient(text, template , model='iodose/nuextract-v1.5'))
    except Exception as e:
        print(f"Error in Structured Text to JSON processing: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"
    

