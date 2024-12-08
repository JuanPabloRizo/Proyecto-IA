# Proyecto-IA
CHATBOT IA
1. Introducción al Proyecto
Este proyecto consistió en la creación de un chatbot inteligente utilizando técnicas de Deep Learning, implementado en Python con la biblioteca TensorFlow. El objetivo principal fue desarrollar un asistente virtual capaz de interactuar con los usuarios de manera fluida, entendiendo sus mensajes y proporcionando respuestas adecuadas basadas en un conjunto de intenciones predefinidas.
2. Objetivo General
Diseñar e implementar un chatbot que permita simular una conversación con el usuario, ofreciendo respuestas automatizadas con un enfoque en flexibilidad y adaptabilidad.
3. Desarrollo del Proyecto

Preparación de los Datos:
Se utilizó un archivo intents.json, que contiene las categorías de intenciones, ejemplos de entrada del usuario y respuestas asociadas. Estos datos fueron el punto de partida para entrenar el modelo.

Preprocesamiento del Lenguaje Natural (NLP):
Utilizando técnicas de procesamiento del lenguaje natural, como la tokenización, lematización y creación de una bolsa de palabras, se prepararon los datos para ser utilizados por el modelo de Deep Learning.

Entrenamiento del Modelo:
El modelo fue diseñado como una red neuronal utilizando TensorFlow, específicamente un modelo secuencial con capas densas. Este modelo fue entrenado para clasificar las intenciones de los mensajes del usuario.

Integración del Modelo:
Se integró el modelo entrenado con una lógica de predicción para interpretar las entradas del usuario y seleccionar las respuestas más adecuadas.

Interfaz de Usuario con Streamlit:
Para hacer el chatbot accesible, se creó una interfaz interactiva utilizando Streamlit. Esta permite al usuario enviar mensajes y recibir respuestas en tiempo real.

