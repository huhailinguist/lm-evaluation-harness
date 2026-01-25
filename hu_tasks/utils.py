def process_docs(dataset):
    # 1. Define your choices and mapping here for consistency
    # The values (0, 1, 2) must match the index of the list in doc_to_choice in YAML
    LABEL_TO_INT = {
        "entailment": 0,    # "对"
        "neutral": 1,       # "不一定"
        "contradiction": 2  # "错"
    }

    # 2. Define your prompt template here
    def create_prompt(premise, hypothesis):
        # You can easily change this string to test new prompts
        return (
            f"请仔细阅读下述信息：\n"
            f"前提：{premise}\n"
            f"问题：依据上述前提，以下判断是否正确？{hypothesis}\n"
            f"选项：对/不一定/错\n"
            f"答案："
        )

    # 3. The processing function applied to every row
    def _process_row(doc):
        # Map the text
        prompt_text = create_prompt(doc["sentence1"], doc["sentence2"])
        
        # Map the label to an integer index
        target_idx = LABEL_TO_INT[doc["label"]]
        
        return {
            "my_custom_prompt": prompt_text,
            "target_index": target_idx
        }

    # 4. Execution Pipeline
    return (
        dataset
        # Filter out bad labels ("-") first
        .filter(lambda x: x["label"] in LABEL_TO_INT)
        # Apply the formatting
        .map(_process_row)
    )