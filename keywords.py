### Test para obtener nuevas keywords ###

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

# Definir el artículo
article = "Este es un ejemplo de artículo que vamos a procesar para extraer keywords. Este artículo tiene algunas palabras clave que deberían aparecer en las keywords."

# Tokenizar el artículo
tokens = word_tokenize(article.lower())

# Eliminar las stopwords
stop_words = set(stopwords.words('spanish'))
tokens = [token for token in tokens if not token in stop_words]

# Calcular la frecuencia de las palabras
fdist = FreqDist(tokens)

# Obtener las palabras más comunes
keywords = [word for word, frequency in fdist.most_common(5)]

print(keywords)