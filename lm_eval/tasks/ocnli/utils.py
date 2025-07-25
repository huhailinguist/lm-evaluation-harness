def process_doc(doc):
    label_mapping = {
        "entailment": "对",
        "neutral": "不一定",
        "contradiction": "错"
    }
    #label '-'
    if doc["label"] not in label_mapping:
        return None
    return {
        "premise": doc["sentence1"],
        "hypothesis": doc["sentence2"],
        "label": doc["label"],
        "choice": label_mapping[doc["label"]],  
        "choices": ["对", "不一定", "错"] 
    }

def process_docs(dataset):
    processed_dataset = dataset.map(process_doc, num_proc=1)
    return processed_dataset.filter(lambda x: x is not None)
