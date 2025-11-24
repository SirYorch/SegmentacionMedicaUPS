## Servidor Flask
El servidor servirá para aplicar -> 

### Denoissing Convolusional neural network

https://github.com/cszn/DnCNN




debemos:



Aplicar:
- Preprocesamiento (blur)
    - gaussiano, para usarlo con blur gaussiano
    
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

- Ecualización de color
    - Ecualizacion de histograma
    - clahe


### final

comparación cuantitativa de métodos clásicos de denoising, con métodos de y DnCNN





# FUNCIONAMIENTO:

- extraer slices de imágenes volumétricas
- Traducir a imagenes manipulables

Toda esta sección se hace al inicio y se guarda en un vector de objetos Mat, con las imagenes. 
tambien podríamos hacerlo directo.

Permitimos un slider para mostrar toda slas imagenes, dando una idea de imagen volumétrica que podemos recorrer hacia atras o adelante 

Una vez con las imagenes, las podemos mostrar con imshow, 


Permitimos un slider para mostrar el tamaño del kernel del blur gaussiano, 

