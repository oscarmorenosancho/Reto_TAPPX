# TAPPX Challenge
## Objective
Association of videos to contextually related articles.

## Algorithm explanation
### 1. Words normalization
We use Spacy to tokenize the text of the document, and also the title if there is one, and the keywords.  The title terms are added to the terms in duplicate so that they have more weight than the text. The keywords are added to the whole terms but also tokenised. On the other hand we also evaluate the coincidence of the classes of the documents, when there is a coincidence we multiply the score of the similarity between documents by 1.2, if they coincide a second time, we multiply again the score by 1.2. That's for increasing the weight of this similarity.
In the tokenisation of the text, title and keywords we also apply a lemmatisation of the nlkt library to each term obtained. In the tokenisation we use the classification in the type of words to eliminate the ones we consider that do not contribute like determiners, prepositions, conjunctions, etc.
### 2. Importance of tokenized words
Getting the string occurrence in each article and each video. The more occurrences, the more unique and important are these words in the document.
### 3. Importance of tokenized words (Full corpus)
Doing the same but with the entire corpus of all articles and videos. The more ocurrences of the word in corpus (all the documents), less relevant it is.
### 4. Vectorisation
In the vectorisation we use the TF-IDF algorithm where we measure the frequency of each term in each document, and the inverse frequency of occurrences in the corpus of documents. Each word in the list of terms in a document gets a TF-IDF measure which is the product of its TF in the document and the IDF in the corpus.
### 5. Score
To calculate the similarity of two documents, we look for matching terms in both documents and calculate the product of the TF-IDF measure they had in each of them.
To obtain the similarity between the documents we add up the scores obtained for each of the matching words, and this is the final similarity score between documents.
### 6. Output
Writing the results in json files.

Note: It's possible to output additional info if the value of "OUTPUT_DBUG" is set to 1. This is the default value.

## Dependencies
nltk 3.8.1
spacy 3.5.1

For spacy we used the model "es_dep_news_trf", which is more accurate and complete version.

## Authors
This project was made by the following students of 42 Barcelona:
* omoreno-
* julolle-
* sersanch
