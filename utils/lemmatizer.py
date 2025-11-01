# Below are resources, needed by nltk
# a simple guardcheck is placed to ensure that
# nltk can function properly

import re, nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk.stem import WordNetLemmatizer

# helper function for importing the necessary packages
def _ensure_nltk():
    need = {
        "tokenizers/punkt": "punkt",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        # tagger name differs across NLTK versions; try both:
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    }
    for handle, pkg in need.items():
        try:
            nltk.data.find(handle)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except:
                pass
_ensure_nltk()

lemmatizer = WordNetLemmatizer()

# pos_tag uses treebank tags. we need to convert them into WordNet tags, for our lemmatizer to understand.
# below is a helper function for that purpose.
def convert_tag(treebank_tag: str):
    if not treebank_tag:
        return NOUN
    tag = treebank_tag[0].upper()
    if tag == 'J':
        return ADJ
    if tag == 'V':
        return VERB
    if tag == 'N':
        return NOUN
    if tag == 'R':
        return ADV
    return NOUN

def preprocess(text: str) -> str:
    text = text.lower() #1. lowercase the whole string
    rext = re.sub(r'[^a-z\s]', '', text) # 2. filter any non-alphabetic characters
    tokens = word_tokenize(text) # 3. split the string, into individual tokens
    tagged_tokens = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tok, convert_tag(tag)) for tok, tag in tagged_tokens if tok.strip()] # each token, we lemmatize (reducing words to their base form)
    return " ".join(lemmas)