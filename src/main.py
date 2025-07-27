### main.py
from rag_engine import answer_question

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")


if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = answer_question(query)
        print(f"Rosa: {response}\n")


