import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()
# palabra = "corriendo"
# palabra2 = "correr"
# palabra3 = "corria"
# raiz = lemmatizer.lemmatize(palabra, pos='v') # 'v' indica que la palabra es un verbo
# raiz2 = lemmatizer.lemmatize(palabra, pos='v') # 'v' indica que la palabra es un verbo
# raiz3 = lemmatizer.lemmatize(palabra, pos='v') # 'v' indica que la palabra es un verbo
# print (raiz, raiz2, raiz3)

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
palabras = ["sumergible", "sumergir", "sol", "solar", "comida", "comiendo", "comer", "computadora", "como", "computar", "la", "el", "de", "a"]
res = []
for palabra in palabras:
    res.append(stemmer.stem(palabra))
print(res)