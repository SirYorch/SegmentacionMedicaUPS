#include "MedicalImageApp.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    // Inicialización de la aplicación Qt
    QApplication a(argc, argv);
    
    // Instancia de la ventana principal
    MedicalImageApp w;
    w.show();

    // Iniciar el bucle de eventos de la aplicación
    return a.exec();
}