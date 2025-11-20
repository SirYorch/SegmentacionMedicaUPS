cmake_minimum_required(VERSION 3.10)
project(MedicalImageProcessor CXX) # Indica que es un proyecto C++

# ==========================================================
# 1. ENCONTRAR MÓDULOS QT
# ==========================================================
# Buscar Qt 6. La búsqueda exitosa de Gui y Widgets implica que Core ya fue encontrado.
find_package(Qt6 COMPONENTS Core Widgets Gui REQUIRED) 

# ==========================================================
# 2. ENCONTRAR ITK y OpenCV
# ==========================================================
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCV REQUIRED)

# ==========================================================
# 3. CONFIGURACIÓN DEL PROYECTO C++ ESTÁNDAR
# ==========================================================

# Usar C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Definir el target
# NO usamos qt_add_executable aquí, usamos add_executable estándar de CMake.
# PERO SÍ necesitamos usar las propiedades de Qt.

add_executable(visor_medico 
    main.cpp 
    MedicalImageApp.cpp 
    MedicalImageApp.h 
    MedicalImageProcessor.cpp 
    MedicalImageProcessor.h
)

# HABILITAR PROPIEDADES DE QT: Esta es la parte crucial que hace lo que la macro hacía
target_link_libraries(visor_medico 
    PRIVATE 
        # Librerías de Qt (las encontramos con find_package)
        Qt6::Core 
        Qt6::Widgets 
        Qt6::Gui
        
        # Librerías de ITK
        ${ITK_LIBRARIES}
        
        # Librerías de OpenCV
        ${OpenCV_LIBS}
)

# Asegurar que el compilador de Meta-Object (MOC) de Qt se ejecute.
# Esto es necesario porque MedicalImageApp.h usa la macro Q_OBJECT.
# La función target_link_libraries(PRIVATE Qt6::Widgets) suele activar esto,
# pero es bueno asegurarse:
set_property(TARGET visor_medico PROPERTY AUTORCC ON) # Para archivos de recursos
set_property(TARGET visor_medico PROPERTY AUTOMOC ON) # Para manejar Q_OBJECT
set_property(TARGET visor_medico PROPERTY AUTOUIC ON) # Para archivos .ui de Qt Designer