def process_doc(doc):
    
    label_map = {"true": "A", "false": "B"}
    correct_letter = label_map[doc["label"].lower()]

    options = ["是", "否"]
    text = doc["text"]
    span1_index = doc["target"]["span1_index"]  
    span2_index = doc["target"]["span2_index"]  
    #Pronoun-index
    span1_text = text[span1_index: span1_index + len(doc["target"]["span1_text"])]
    span2_text = text[span2_index: span2_index + len(doc["target"]["span2_text"])]
    return {
        "text": text,
        "span1_text": span1_text,
        "span2_text": span2_text,
        "options": options,
        "label": correct_letter
    }
    
def process_docs(dataset):
    return dataset.map(process_doc, num_proc=1)
    

