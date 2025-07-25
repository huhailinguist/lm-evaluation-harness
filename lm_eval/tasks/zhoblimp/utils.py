import random

def process_doc(doc):
    # seed
    seed = hash(doc["UID"] + str(doc["pairID"])) % 2**32
    random.seed(seed)
    
    choices = [doc["sentence_good"], doc["sentence_bad"]]
    if random.random() < 0.5:
        choices = [doc["sentence_bad"], doc["sentence_good"]]
        correct_answer = 1  
    else:
        correct_answer = 0  
    
    return {
        "UID": doc["UID"],
        "phenomenon": doc["phenomenon"],
        "pairID": doc["pairID"],
        "choices": choices,
        "correct_answer": correct_answer
    }

def process_docs(dataset):
    return dataset.map(process_doc, num_proc=1)
