def process_docs(dataset):
    # 1. Define the mapping
    label_mapping = {
        "entailment": 0,    # Corresponds to index 0 in doc_to_choice
        "neutral": 1,       # Corresponds to index 1
        "contradiction": 2  # Corresponds to index 2
    }

    # 2. Filter out bad rows (label "-") FIRST
    # We check if the label exists in our mapping keys
    dataset = dataset.filter(lambda x: x["label"] in label_mapping)

    # 3. Map to rename columns and format targets
    def _map_fn(doc):
        return {
            "premise": doc["sentence1"],
            "hypothesis": doc["sentence2"],
            # We return the INTEGER index of the answer (0, 1, or 2)
            # This is safer than returning strings
            "label_index": label_mapping[doc["label"]] 
        }

    return dataset.map(_map_fn)