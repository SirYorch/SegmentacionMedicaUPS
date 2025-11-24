## Servidor Flask
El servidor servirá para aplicar -> 

### Denoissing Convolusional neural network

https://github.com/cszn/DnCNN




debemos:

- extraer slices de imágenes volumétricas
- Traducir a imagenes manipulables

Aplicar:
- Preprocesamiento
    - Ecualizacion
    - clahe
- Binarización
    - umbral
- Detección de bordes

- contraste
- identificación de regiones

- reducción de ruido
    - mediana
    - media
    - gaussiana
- operaciones morfológicas
    - erosión
    - dilatación



### final

comparación cuantitativa de métodos clásicos de denoising, con métodos de y DnCNN









# Verificar que todo esté instalado
make check

# Ver información de compilación
make info

# Compilar el proyecto
make

# Ejecutar
make run

# O ejecutar directamente
./mi_app