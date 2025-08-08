#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para procesar los resultados de un batch de OpenAI en formato JSONL,
extraer la información relevante y fusionarla con un archivo CSV de entrada.

Las rutas de los archivos de entrada y salida están predeterminadas en el código.
"""

import pandas as pd
import json
import sys
import os

# --- ⚙️ CONFIGURACIÓN DE RUTAS ---
# Modifica estas variables para cambiar los archivos de entrada o salida.
JSONL_INPUT_PATH = "data/json/batch_vwx6bfxuDSrlSctyJrMgionx_output.jsonl"
CSV_INPUT_PATH = "data/clean/delitos_imagenes_santiago_limpio.csv"
CSV_OUTPUT_PATH = 'data/inferences/delitos_imagenes_santiago_limpio_con_descripcion.csv'
# ---------------------------------

def process_jsonl_data(jsonl_path: str) -> pd.DataFrame:
    """
    Lee un archivo JSONL, extrae datos específicos de cada línea y los devuelve en un DataFrame.

    Args:
        jsonl_path (str): La ruta al archivo de entrada .jsonl.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas 'index', 'result' y 'raw_response'.
    """
    results_list = []
    print(f"📄 Procesando el archivo JSONL: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extrae el índice numérico del 'custom_id' (ej: 'task-123' -> 123)
                task_id = data['custom_id']
                index = int(task_id.split('-')[-1])
                
                # Extrae el contenido del mensaje de la respuesta
                result = data['response']['body']['choices'][0]['message']['content']
                
                results_list.append({
                    'index': index,
                    'result': result,
                    'raw_response': data  # Guarda la respuesta original completa por si se necesita
                })
            except (KeyError, IndexError, TypeError) as e:
                # Si una línea no tiene la estructura esperada, lo notifica y continúa
                print(f"⚠️ Advertencia: No se pudo procesar la línea para el ID '{data.get('custom_id', 'N/A')}'. Error: {e}")
            except json.JSONDecodeError:
                print(f"⚠️ Advertencia: Se encontró una línea que no es un JSON válido y fue omitida.")

    if not results_list:
        print("❌ Error: No se pudo extraer ningún dato válido del archivo JSONL.")
        sys.exit(1) # Termina el script si no hay datos

    print(f"✅ Se procesaron {len(results_list)} resultados del archivo JSONL.")
    return pd.DataFrame(results_list)

def main():
    """Función principal del script."""
    print("--- Iniciando Proceso de Fusión de Datos ---")
    
    # 1. Procesar los resultados del archivo JSONL
    try:
        results_df = process_jsonl_data(JSONL_INPUT_PATH)
    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada JSONL no fue encontrado en '{JSONL_INPUT_PATH}'")
        sys.exit(1)

    # 2. Cargar el DataFrame original
    try:
        main_df = pd.read_csv(CSV_INPUT_PATH)
        print(f"📖 Cargado el archivo CSV principal: {CSV_INPUT_PATH} ({len(main_df)} filas)")
    except FileNotFoundError:
        print(f"❌ Error: El archivo de entrada CSV no fue encontrado en '{CSV_INPUT_PATH}'")
        sys.exit(1)

    # 3. Fusionar los DataFrames
    # Se usa 'left_index=True' en el DF original y 'right_on='index'' en el de resultados.
    # Esto asume que el 'index' extraído del 'custom_id' corresponde al índice de la fila en el CSV original.
    print("🔄 Fusionando los datos...")
    final_df = main_df.merge(
        results_df,
        left_index=True,
        right_on='index',
        how='left'
    )
    
    # Opcional: Eliminar la columna 'index' si ya no es necesaria
    if 'index' in final_df.columns:
        final_df = final_df.drop(columns=['index'])

    # 4. Asegurarse de que el directorio de salida exista
    output_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if output_dir: # Si no es una ruta en el directorio actual
        os.makedirs(output_dir, exist_ok=True)

    # 5. Guardar el resultado final
    final_df.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8')
    print("---------------------------------------------")
    print(f"🎉 ¡Éxito! Archivo final guardado en: {CSV_OUTPUT_PATH}")
    print("\nPrimeras 5 filas del resultado:")
    print(final_df.head())

if __name__ == "__main__":
    main