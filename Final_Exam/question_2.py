import spacy
from spacy.language import Language

@Language.component("custom_component")
def custom_component(doc):
    # Custom component that simply prints out the token and the lemma of the token
    for token in doc:
        print(f"Token Text: {token.text}\t Token Lemma {token.lemma_}")
    return doc

# Load the English language model with only the Tokenizer
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

# Add the custom component to the pipeline
nlp.add_pipe("custom_component", name="custom_component", last=True)

# Process a text
text = "This is an example sentence."
doc = nlp(text)
