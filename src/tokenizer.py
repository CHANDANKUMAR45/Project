"""
tokenizer.py

Custom tokenizer for Topic Classification.
.


"""

import re


class Tokenizer:

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean input text.
        """

        text = str(text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove Emails
        text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML Tags
        text = re.sub(r"<.*?>", "", text)

        # Remove Numbers
        text = re.sub(r"\d+", "", text)

        # Keep only alphabets
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize(self, text: str):
        """
        Convert sentence into tokens.
        """

        cleaned = self.clean_text(text)

        return cleaned.split()


if __name__ == "__main__":

    tokenizer = Tokenizer()

    sample = """
    Hello!!! Visit https://openai.com.
    Email me at abc@gmail.com
    Machine Learning 2026 is Awesome!!!
    """

    print("Original Text:\n")
    print(sample)

    print("\nTokens:\n")
    print(tokenizer.tokenize(sample))