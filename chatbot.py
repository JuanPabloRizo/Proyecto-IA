import random   
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Inicialización de lematizador
lemmatizer = WordNetLemmatizer()

# Cargar los datos
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Convierte la oración a una lista de palabras lematizadas"""
    sentence_words = nltk.word_tokenize(sentence)  # Tokeniza la oración
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lematiza las palabras
    return sentence_words

def bag_of_words(sentence):
    """Convierte la oración en una bolsa de palabras binaria (1 o 0 para cada palabra en el vocabulario)"""
    sentence_words = clean_up_sentence(sentence)  # Limpia la oración
    bag = [0] * len(words)  # Inicializa la bolsa de palabras con ceros
    for w in sentence_words:  # Recorre las palabras de la oración
        for i, word in enumerate(words):  # Compara con el vocabulario
            if word == w:
                bag[i] = 1  # Marca 1 si la palabra está presente
    return np.array(bag)  # Devuelve la bolsa de palabras como un arreglo numpy

def predict_class(sentence):
    """Predice la clase de la intención de la oración dada"""
    bow = bag_of_words(sentence)  # Crea una bolsa de palabras a partir de la entrada del usuario
    res = model.predict(np.array([bow]))[0]  # Predice la clase de la entrada del usuario
    ERROR_THRESHOLD = 0.25  # Umbral de error (25%)
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtra los resultados por el umbral
    results.sort(key=lambda x: x[1], reverse=True)  # Ordena los resultados por la probabilidad
    return_list = []  # Lista vacía para almacenar los resultados
    for r in results:  # Recorre los resultados
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Añade la intención y probabilidad
    return return_list  # Devuelve la lista de intenciones y probabilidades

def get_response(intents_list, intents_json):
    """Obtiene una respuesta basada en la intención predicha"""
    tag = intents_list[0]['intent']  # Obtiene la intención predicha
    list_of_intents = intents_json['intents']  # Obtiene la lista de intenciones definidas
    for i in list_of_intents:  # Busca la intención correspondiente
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Escoge una respuesta aleatoria de las opciones
            break
    return result  # Devuelve la respuesta

# Función para interactuar con el chatbot
def chatbot_response(text):
    """Función principal para obtener una respuesta del chatbot"""
    intents_list = predict_class(text)  # Predice la clase de la intención
    if intents_list:
        response = get_response(intents_list, intents)  # Obtiene la respuesta basada en la intención
    else:
        response = "Lo siento, no te entendí. ¿Podrías reformular la pregunta?"  # Respuesta predeterminada
    return response  # Devuelve la respuesta final
