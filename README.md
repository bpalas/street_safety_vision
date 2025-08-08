# street_safety_vision-



# Street Safety Vision

## Descripción

Este proyecto utiliza modelos de visión por computadora para analizar la seguridad en las calles a gran escala. Se aprovecha la API de OpenAI por lotes (batch) para realizar inferencias de manera económica y rápida.

## Estructura del Proyecto

  - **/notebooks**: Contiene los notebooks `01` y `02` que se utilizaron para pruebas iniciales y experimentación.
  - **/src**: Contiene los scripts principales para la inferencia a gran escala.

## Metodología

Para el análisis de imágenes a gran escala, se implementó una solución que utiliza la API de OpenAI por lotes. Dado que la API no permite el envío directo de imágenes, estas se almacenan en un bucket de Google Cloud Storage (GCS). Luego, se envían las URLs de las imágenes en las solicitudes a la API.

Este enfoque ha demostrado ser:

  - **Económico**: Con un costo aproximado de 4-5 dólares por comuna.
  - **Rápido**: El procesamiento de una comuna completa se realiza en menos de 24 horas.

Un ejemplo de una URL de imagen almacenada en GCS es el siguiente:

```
https://storage.googleapis.com/ciudades-crimen/googlemaps/-33.53283901577287,-70.59290969400631.jpg
```

## Modelos Utilizados

Es importante destacar que los modelos utilizados en este proyecto se encuentran depreciados:

  - **phi-3-vision**: Modelo gratuito que ya no está disponible.
  - **Modelo de OpenAI (2024)**: El modelo específico de OpenAI utilizado para la inferencia en 2024 también ha sido depreciado.

-----