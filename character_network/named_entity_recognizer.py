import pandas as pd
import spacy
import nltk
from nltk import sent_tokenize
from ast import literal_eval
import os
import sys
import pathlib
from utils import load_subtitles_dataset

# Download NLTK sentence tokenizer data if not already present
nltk.download('punkt', quiet=True)


class NamedEntityRecognizer:
    def __init__(self, model_name="en_core_web_sm"):
        self.model_name = model_name
        self.nlp_model = self.load_model()

    def load_model(self):  # Added self parameter
        """Load the spaCy NLP model"""
        try:
            nlp = spacy.load(self.model_name)
            return nlp
        except OSError:
            raise ValueError(f"Model {self.model_name} not found. Please install it first.")

    def get_ners_inference(self, script):
        """Extract named entities from a script"""
        script_sentences = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text.strip()
                    first_name = full_name.split()[0] if full_name else ""
                    if first_name:  # Only add non-empty names
                        ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        """Process dataset to extract named entities"""

        # Load from cache if available
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load the dataset
        df = load_subtitles_dataset(dataset_path)

        # Run NER inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        # Save results if requested
        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df