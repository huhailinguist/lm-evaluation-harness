# utils.py

LABEL_TO_INT = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

def _build_process_docs(create_prompt):
    """Factory that returns a process_docs function for a given prompt template."""
    def process_docs(dataset):
        def _process_row(doc):
            prompt_text = create_prompt(doc["sentence1"], doc["sentence2"])
            target_idx = LABEL_TO_INT[doc["label"]]
            return {
                "my_custom_prompt": prompt_text,
                "target_index": target_idx,
            }

        return (
            dataset
            .filter(lambda x: x["label"] in LABEL_TO_INT)
            .map(_process_row)
        )
    return process_docs


# --- Prompt Style A: 推理关系 ---
def _prompt_a(premise, hypothesis):
    return (
        f"请判断句子一能否推理得到句子二：\n"
        f"句子一：{premise}\n"
        f"句子二：{hypothesis}\n"
        f"答案："
    )

# --- Prompt Style B: 是否成立 ---
def _prompt_b(premise, hypothesis):
    return (
        f"已知：{premise}\n"
        f"请问，下面这句话是否成立：{hypothesis}\n"
        f"答案："
    )

# --- Prompt Style C: 正确/错误/都有可能 ---
def _prompt_c(premise, hypothesis):
    return (
        f"{premise}\n问题：{hypothesis}\n正确、错误，还是都有可能？\n答案："
    )


# These are the three entry points your YAMLs will reference
process_docs = _build_process_docs(_prompt_a)
process_docs2 = _build_process_docs(_prompt_b)
process_docs3 = _build_process_docs(_prompt_c)