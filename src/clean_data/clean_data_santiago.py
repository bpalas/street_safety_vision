import pandas as pd
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv

def main():
    # Cargar datos de imágenes
    df2 = pd.read_csv('data/raw/urls_imagenes.csv')  # CSV de imágenes

    # Cargar datos de delitos
    df = pd.read_csv('data/raw/delitos_imagenes_santiago_traducido.csv')  # CSV de delitos

    # Eliminar duplicados y valores nulos en 'nombre_foto'
    df = df.drop_duplicates(subset=['nombre_foto'])
    df = df.dropna(subset=['nombre_foto'])

    # Paso 1: Eliminar '/content/' de 'file_name' en df2
    df2['nombre_foto'] = df2['file_name'].str.replace('/content/', '', regex=False)

    # Paso 2: Realizar el merge en base a 'nombre_foto'
    df = df.merge(df2[['nombre_foto', 'public_url']], on='nombre_foto', how='left')

    # Guardar el resultado en un nuevo archivo CSV
    df.to_csv('data/clean/delitos_imagenes_santiago_limpio.csv', index=False)
    print("Archivo procesado y guardado en 'data/clean/delitos_imagenes_santiago_limpio.csv'.")

if __name__ == "__main__":

        main()
        