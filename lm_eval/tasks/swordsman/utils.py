
def process_doc(doc):
    correct_letter = doc["task1"]["correct_answer"]
    options = doc["task1"]["options"]
    return {
        "dialogue": doc["dialogue"],
        "character": doc["character"],  
        "utterance": doc["utterance"], 
        "options": options,
        "label": correct_letter 
    }

def process_docs(dataset):
    return dataset.map(process_doc, num_proc=1)
