from typing import Dict, List, Union
from gliner import GLiNER

def extract_pii(
    text: str, 
    labels: str = None, 
    threshold: float = 0.5, 
    nested_ner: bool = False
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Extract personally identifiable information (PII) from text using GLiNER model.
    
    Args:
        text (str): The text to analyze for PII
        labels (str): Comma-separated list of labels to look for (if None, uses default labels)
        threshold (float): Confidence threshold for entity detection (0.0 to 1.0)
        nested_ner (bool): Whether to allow nested entity recognition
    
    Returns:
        Dict containing the original text and a list of detected entities
    """
    # Initialize model if not already done
    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    
    # Default PII labels if none provided
    if labels is None:
        labels = [
            "person", "organization", "address", "email", "phone number", 
            "social security number", "credit card number", "passport number", 
            "driver license", "bank account number", "date of birth", 
            "medical record number", "insurance policy number", "property registration number",
            "employee ID number", "tax ID number", "full address", "personally identifiable information"
        ]
    else:
        labels = [label.strip() for label in labels.split(",")]
    
    # Extract entities
    entities = model.predict_entities(
        text, 
        labels, 
        flat_ner=not nested_ner, 
        threshold=threshold
    )
    
    # Format results
    results = {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": entity.get("score", 0),
            }
            for entity in entities
        ],
    }
    
    return results


# Example usage
if __name__ == "__main__":
    sample_text = "John Smith, from London, teaches mathematics at Royal Academy located at 25 King's Road. His employee ID is UK-987654-321 and he has been working there since 2015."
    
    # Example 1: Using default labels
    results = extract_pii(sample_text)
    print("Example 1: Using default labels")
    print(f"Text: {results['text']}")
    print("Detected entities:")
    for entity in results["entities"]:
        print(f"  {entity['word']} => {entity['entity']} (positions {entity['start']}-{entity['end']})")
    
    # Example 2: Using custom labels with different threshold
    custom_labels = "person, profession, organization, address, employee ID number"
    results = extract_pii(sample_text, labels=custom_labels, threshold=0.3)
    print("\nExample 2: Using custom labels with lower threshold")
    print(f"Text: {results['text']}")
    print("Detected entities:")
    for entity in results["entities"]:
        print(f"  {entity['word']} => {entity['entity']} (positions {entity['start']}-{entity['end']})")