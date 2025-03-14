# Fundamentos de procesamiento de datos para Data Science

# Implementación Agente AI

## Nombre del grupo:

* ESESA 2024/2025 TFM Grupo 3 - Máster en BigData e Ingeniería de Datos

## Integrantes:

* [Pedro Acosta](https://github.com/poacosta)
* [Francisco Salas](https://github.com/Firulais730)
* [Rushabh Patel](https://github.com/rushabhp01)

# Documentación

## Descripción del agente

El VideoClub Agent ESESA es un asistente conversacional especializado en cine que combina la API de OpenAI con una base
de datos PostgreSQL para proporcionar información detallada sobre películas y actores.

El agente funciona como un experto en cine que:

- Responde preguntas sobre películas, actores y temas cinematográficos
- Consulta una base de datos real para obtener información precisa
- Proporciona opiniones fundamentadas sobre películas y sus características
- Mantiene un tono conversacional y amigable como un experto en videoclub

El flujo de trabajo del sistema funciona de la siguiente manera:

1. El usuario formula una pregunta sobre cine
2. El sistema utiliza el LLM para identificar si la consulta es sobre una película o un actor
3. Si se identifica una entidad, consulta la base de datos PostgreSQL
4. Combina los resultados de la base de datos con el conocimiento general del modelo (LLM)
5. Genera una respuesta natural que incorpora tanto los datos estructurados como el conocimiento contextual

## Cómo ejecutarlo

### Requisitos previos

- Python 3.10+
- Una clave de API de OpenAI
- Acceso a la base de datos PostgreSQL (Base de datos proporcionada para el ejercicio)
- Las siguientes bibliotecas instaladas:
    - gradio
    - openai
    - psycopg2
    - python-dotenv

### Configuración

1. Clona este repositorio en tu máquina local
2. Instala las dependencias necesarias:
   ```
   pip install -r requirements.txt
   ```
3. Crea un archivo `.env` en el directorio raíz con las siguientes variables:
   ```
   OPENAI_API_KEY=tu_clave_api_openai
    # Estos datos se comparten dado que la base de datos es de prueba y tiene control por IP
    # No haga esto en un ambiente de producción
    DB_HOST=217.71.200.156
    DB_PORT=5433
    DB_NAME=esesa
    DB_USER=esesa
    DB_PASSWORD=esesa
    # Estos datos se comparten dado que la base de datos es de prueba y tiene control por IP
   ```
   (Nota: Los valores de base de datos mostrados son los predeterminados y deberían modificarse en un entorno de
   producción)

### Ejecución

1. Abre una terminal en el directorio del proyecto
2. Ejecuta el siguiente comando:
   ```
   python agent.py
   ```
3. Se iniciará un servidor Gradio y se mostrará una URL local
4. Abre esta URL en tu navegador para interactuar con el agente
5. Opcionalmente, puedes utilizar el enlace público temporal proporcionado por Gradio para compartir tu agente

### Ejemplos de consultas

**Nota importante:** Para obtener mejores resultados, se recomienda chatear con el agente en inglés, ya que el contenido
de la base de datos está en inglés.

Ejemplos de consultas recomendadas:

- Do you have the movie Agent Truman?
- Tell me about Alabama Devil
- Who was the actor/actress in Angels Life?

### Interfaz de usuario

* La aplicación web de Gradio proporciona una interfaz de usuario simple y fácil de usar. Puedes escribir preguntas en
  el
  cuadro de texto y ver las respuestas generadas por el agente.

### Base de datos

* La base de datos sigue este esquema:


