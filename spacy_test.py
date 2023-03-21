import spacy

# Cargar el modelo de idioma español
nlp = spacy.load("es_core_news_sm")

# Procesar el texto
doc = nlp("El perro marrón saltó sobre el zorro perezoso.")

# Imprimir las palabras y sus etiquetas de partes del habla
for token in doc:
    print(token.text, token.pos_)
