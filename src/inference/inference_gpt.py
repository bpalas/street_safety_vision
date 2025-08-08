# -*- coding: utf-8 -*-

"""
Script para analizar imágenes en lote (batch) usando la API de Visión de OpenAI.

Este script realiza los siguientes pasos:
1.  Define y crea la estructura de carpetas necesaria.
2.  Carga un prompt de sistema desde un archivo externo.
3.  Lee un archivo CSV que contiene las URLs públicas de las imágenes a analizar.
4.  Genera una tarea para cada imagen, asignando un ID único.
5.  Crea un archivo de batch en memoria (sin guardarlo en disco).
6.  Pide confirmación al usuario antes de enviar el lote para controlar costos.
7.  Sube el archivo en memoria y crea el trabajo de batch en OpenAI.
"""

# --- Parte 1: Configuración e Inicialización ---
import json
import io
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Carga variables de entorno desde un archivo .env en la raíz del proyecto
load_dotenv()

# --- Configuración de Rutas del Proyecto ---
# El script está en 'src/inference/', así que subimos DOS niveles para llegar a la raíz.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ruta al archivo que contiene el prompt del sistema.
PROMPT_FILE_PATH = PROJECT_ROOT / 'prompts' / 'caption_system_prompt.txt'

# Ruta al archivo CSV limpio que contiene las URLs de las imágenes.
INPUT_CSV_PATH = PROJECT_ROOT / 'data' / 'clean' / 'delitos_imagenes_santiago_limpio.csv'

# Carpeta de salida para los resultados que la API de OpenAI generará.
# Este script no crea archivos aquí, pero es una buena práctica tenerla definida.
BATCH_OUTPUT_FOLDER_PATH = PROJECT_ROOT / 'data' / 'inferences'

# --- Creación de Carpetas ---
# Asegura que las carpetas necesarias existan.
BATCH_OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'prompts').mkdir(parents=True, exist_ok=True)

# --- Funciones Auxiliares ---
def load_prompt_from_file(file_path: Path) -> str:
    """Carga y devuelve el contenido de un archivo de texto."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error CRÍTICO: El archivo de prompt '{file_path}' no se encontró.")
        raise
    except Exception as e:
        print(f"❌ Error CRÍTICO al leer el archivo de prompt '{file_path}': {e}")
        raise

# --- Inicialización del Cliente y Carga del Prompt ---
try:
    client = OpenAI()
    print("✅ Cliente de OpenAI inicializado correctamente.")
except Exception as e:
    print(f"❌ Error CRÍTICO al inicializar el cliente de OpenAI: {e}")
    raise

SYSTEM_PROMPT = load_prompt_from_file(PROMPT_FILE_PATH)
print("✅ Prompt del sistema cargado.")


# --- Parte 2: Carga de Datos y Generación de Tareas ---
print("\nIniciando la generación de tareas para el batch de imágenes...")

try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"✅ Se cargaron {len(df)} registros desde '{INPUT_CSV_PATH.name}'.")
except FileNotFoundError:
    print(f"❌ Error CRÍTICO: No se encontró el archivo de entrada en '{INPUT_CSV_PATH}'.")
    raise
except Exception as e:
    print(f"❌ Error CRÍTICO al leer el archivo CSV: {e}")
    raise

batch_tasks_list = []
for index, row in df.iterrows():
    # Asumimos que el CSV tiene columnas 'nombre_foto' y 'public_url'
    image_id = row.get('nombre_foto', f'task-{index}') # Usa 'nombre_foto' si existe, si no un genérico
    image_url = row.get('public_url')

    if not image_url:
        print(f"⚠️ Advertencia: Fila {index} no tiene 'public_url'. Saltando.")
        continue

    # Crea el cuerpo de la petición para cada imagen
    task_item = {
        "custom_id": str(image_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini", # Modelo optimizado y más económico
            "temperature": 0.2,
            "max_tokens": 500, # Ajusta según la longitud de respuesta esperada
            "response_format": {"type": "json_object"}, # Recomendado si esperas un JSON
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analiza la siguiente imagen de una calle y describe los elementos relevantes según tus instrucciones."},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}} # detail: low para ahorrar tokens
                    ],
                }
            ]
        }
    }
    batch_tasks_list.append(task_item)

print(f"✅ Generación de tareas completada. Total de tareas: {len(batch_tasks_list)}")

# --- Parte 3: Creación y Envío del Archivo Batch ---
if not batch_tasks_list:
    print("No se generaron tareas. Finalizando el proceso.")
else:
    # Confirmación del usuario para evitar costos inesperados
    user_confirmation = input(f"Se han generado {len(batch_tasks_list)} tareas. ¿Deseas enviar el batch a OpenAI? (s/N): ")

    if user_confirmation.lower() != 's':
        print("Envío cancelado por el usuario.")
    else:
        print("\n⚙️  Procediendo con la creación y envío del batch...")

        try:
            # Crear el archivo JSONL en memoria para no guardarlo en disco
            batch_stream = io.BytesIO()
            for task in batch_tasks_list:
                batch_stream.write((json.dumps(task) + '\n').encode('utf-8'))
            batch_stream.seek(0) # Regresar al inicio del stream para que la API lo lea

            # 1. Subir el archivo en memoria a OpenAI
            print("Subiendo archivo batch a OpenAI...")
            batch_file = client.files.create(
                file=("batch_input.jsonl", batch_stream), # Pasamos el stream como un archivo
                purpose="batch"
            )
            print(f"✅ Archivo subido con ID: {batch_file.id}")

            # 2. Crear el trabajo de batch
            print("Creando el job de batch en OpenAI...")
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(f"🚀 ¡Éxito! Job de batch enviado con ID: {batch_job.id}")
            print("Puedes monitorear el estado en tu dashboard de OpenAI o con otro script.")

        except Exception as e:
            print(f"❌ Error CRÍTICO durante el envío del batch a OpenAI: {e}")