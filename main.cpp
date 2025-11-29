#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkGDCMImageIO.h>

using namespace std;
using namespace cv;
using namespace itk;


// ----- VARIABLES GLOBALES

using InputPixelType  = short;
using OutputPixelType = unsigned char;
using InputImageType  = Image<InputPixelType, 2>;
using OutputImageType = Image<OutputPixelType, 2>;

int currentSlice = 0;
int ksizeTrack = 3;       
int sigmaTrack = 10;      
int umbralMin = 0;
int umbralMax = 255;
int cx = 100;
int cy = 100;
int radioX = 50;
int radioY = 50;

int tiempo = 0;

int kernelSize = 3;

bool clahe = false;
bool eq = false;

bool controles = false;
bool slicer = false;

vector<Mat> imgs;



struct Button {
    int x1, y1, x2, y2;
};

 
Button btn_corazon  = {50, 225, 147, 252};
Button btn_huesos   = {200, 225, 300, 252};
Button btn_pulmones = {359, 225, 460, 252};

Button btn_result   = {50, 297, 147, 330};
Button btn_comp     = {200, 297, 300, 330};
Button btn_extra    = {359, 297, 460, 330};

Button btn_pruebas    = {200, 345, 300, 375};

bool corazon = false;
bool hueso= false;
bool pulmones= false;
bool result= false;
bool comparacion= false;
bool extra= false;
bool pruebas= false;


Mat img_normal; // imagen de menu
Mat img_click;  // imagen de menu cuando se cliquea para mostrar la máscara



// Declaración global de botones de la APP (no del menú) -- aplicación de ecualización

Rect btnCLAHE(10, 10, 120, 40);
Rect btnEq(10, 60, 120, 40);

// ... [TUS FUNCIONES AUXILIARES: ITKToMat, readRaw, readIMA, etc. SE MANTIENEN IGUAL] ...
// (Las pego colapsadas para ahorrar espacio, el código lógico no cambia)

void updateKSize(int KSize){ KSize = KSize*2+1; }



// método para obtener una lista de todos los archivos IMA de la carpeta L333 del datasetKaggle
vector<string> getIMA(const string& dir) {
    vector<string> files;
    for (auto& e : filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        string ext = e.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".ima") files.push_back(e.path().string());
    }
    sort(files.begin(), files.end());
    return files;
}


// método para leer imagenes .ima y transformarmas a objetos Mat de opencv para manipularlas

Mat readIMA(const string& filename)
{
    // --- Configurar lector DICOM ---
    using ImageIOType = GDCMImageIO;
    ImageIOType::Pointer dicomIO = ImageIOType::New();

    using ReaderType = ImageFileReader<InputImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileName(filename);
    reader->Update();

    // --- Rescalar intensidades a 0..255 ---
    using RescaleType = RescaleIntensityImageFilter<InputImageType, OutputImageType>;
    RescaleType::Pointer scale = RescaleType::New();
    scale->SetInput(reader->GetOutput());
    scale->SetOutputMinimum(0);
    scale->SetOutputMaximum(255);
    scale->Update();

    OutputImageType::Pointer img = scale->GetOutput();

    // --- Obtener tamaño ---
    auto region   = img->GetLargestPossibleRegion();
    auto size     = region.GetSize();
    int width     = size[0];
    int height    = size[1];

    // --- Crear Mat ---
    Mat out(height, width, CV_8UC1);

    // --- Copiar buffer ITK → OpenCV ---
    unsigned char* buffer = img->GetBufferPointer();
    memcpy(out.data, buffer, width * height);

    return out;
}



// método para activar filtro CLAHE en pruebas
Mat toClahe(Mat imagen){
    Mat salida;
    Ptr<CLAHE> clahe = createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(imagen, salida);
    return salida;
}

// método para activar Ecualización en pruebas
Mat toEq(Mat imagen){
    equalizeHist(imagen, imagen);
    return imagen;
}

// método para Filtro de Sharpening

Mat applySharpening(Mat input) {
    Mat output = input.clone();

    float g = 1.66f;
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, g) * 255.0);
    LUT(output, lookUpTable, output);

    double sigma = 2.8; 
    double amount = 1.2;

    Mat blurred;
    GaussianBlur(output, blurred, cv::Size(0, 0), sigma);
    addWeighted(output, 1.0 + amount, blurred, -amount, 0, output);

    return output;
}


// método para filtro para rellenar zonas vacías
Mat fillHoles(const Mat& mask) {
    Mat im_floodfill = mask.clone();
    Mat maskFlood;
    copyMakeBorder(im_floodfill, maskFlood, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    floodFill(maskFlood, cv::Point(0, 0), Scalar(255));
    Mat floodfilled = maskFlood(Rect(1, 1, mask.cols, mask.rows));
    Mat floodfilledInv;
    bitwise_not(floodfilled, floodfilledInv);
    return (mask | floodfilledInv);
}






// método para hacer filtro de umbral, aqui creamos una máscara que considere valores máximos y minimos de intensidad
Mat Umbrilize(Mat imagen, int umbralMin, int umbralMax){
    Mat img = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
    for(int i =0 ; i < imagen.rows ; i ++){
        for(int j =0 ; j < imagen.cols ; j ++){
            uchar pixel = imagen.at<uchar>(i, j);
            if (pixel >= umbralMin && pixel <= umbralMax) img.at<uchar>(i, j) = 255;     
            else img.at<uchar>(i, j) = 0;       
        }   
    }
    return img;
}

// método para activar Blur gaussiano, y cambiar los valores del kernel y sigma en la ecuacion 
// $exp(-(x+y)/(2*sigma^2))$

Mat toGaussianBlur(Mat imagen, int kernelSize = 5, double sigma = 0) {
    Mat salida;
    if (kernelSize % 2 == 0) kernelSize++;
    GaussianBlur(imagen, salida, cv::Size(kernelSize, kernelSize), sigma);
    return salida;
}




// filtro para seleccionar zona de interés en huesos, eliminar estomago
Mat filterByAreaAndIntensity(const Mat& mask, const Mat& original) {
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    Mat out = Mat::zeros(mask.size(), CV_8UC1);

    for (int label = 1; label < nLabels; ++label) {
        int area = stats.at<int>(label, CC_STAT_AREA);

        if (area < 57) continue;

        if (currentSlice >= 245 && currentSlice <= 438) {
            
            double cX = centroids.at<double>(label, 0);
            double cY = centroids.at<double>(label, 1);
            double termX = std::pow(cX - 260, 2) / std::pow(127.0, 2);
            double termY = std::pow(cY - 217, 2) / std::pow(94.0, 2);

            if ((termX + termY) <= 1.0) {
                if (area < 13187) {
                    continue; 
                }
            }
        }

        Mat componentMask = (labels == label);
        Scalar meanVal = mean(original, componentMask);
        if (meanVal[0] < 83) continue;

        out.setTo(255, componentMask);
    }
    return out;
}


// método para Generar un circulo de máscara, nos sirve para tomar los valores que deseamos e ignorar lo demás, basicamente 
// marcar la región de interés
Mat maskCircle(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);
    Mat salida(img.rows, img.cols, img.type(), Scalar(255));
    
    img.copyTo(salida, mask);
    return salida;
}
Mat maskCircle2(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);

    // 1. Dibujar círculo blanco = zona visible
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);

    // 2. Crear salida blanca (color blanco completo)   
    Mat salida = Mat::zeros(img.size(), img.type());

    // 3. Copiar dentro del círculo la imagen original
    img.copyTo(salida, mask);

    return salida;
}
// método para hacer erosión y dilatación
Mat open(Mat imagen, int kernelSize){
    Mat erodida, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    erode(imagen, erodida, kernel);
    dilate(erodida, salida, kernel);
    return salida;
}

// método para hacer dilatación y erosión
Mat close(Mat imagen, int kernelSize){
    Mat dilatada, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    dilate(imagen, dilatada, kernel);
    erode(dilatada, salida, kernel);
    return salida;
}


//  definición de métodos, para poder usarlos previo a su creación

Mat mejorarNitidez(Mat imagenEntrada);



//MÉTODO PARA LA "interfaz gráfica" para saber si el click se dio adentro o fuera de un botón

bool inside(Button b, int x, int y) {
    return (x > b.x1 && x < b.x2 && y > b.y1 && y < b.y2);
}


// método para dibujar la interfaz cuando se presiona un botón, permite marcar en donde se presionó generando una máscara

void drawMask(Mat &img, Button b) {
    Rect rect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

    // Crear una copia de la imagen original para el overlay
    Mat overlay;
    img.copyTo(overlay);

    // Color del rectángulo (BGR)
    Scalar color(255, 100, 100); // Azul suave
    double alpha = 0.2; // 20% de opacidad

    // Dibujar el rectángulo lleno en el overlay
    rectangle(overlay, rect, color, FILLED);

    // Mezclar overlay + original según la máscara del rectángulo
    addWeighted(overlay, alpha, img, 1 - alpha, 0, img);
}


void encenderVentana(int boton){
    tiempo = 1;
    corazon = false;
    hueso= false;
    pulmones= false;
    result= false;
    comparacion= false;
    extra= false;
    pruebas= false;

    if(boton  == 1){
        corazon = true;
    } else if(boton  == 2){
        hueso = true;
    } else if(boton  == 3){
        pulmones = true;
    } else if(boton  == 4){
        result = true;
    } else if(boton  == 5){
        comparacion = true;
    } else if(boton  == 6){
        extra = true;
    } else if(boton  == 7){
        pruebas = true;
    }
}

// --- EVENTOS DE CLICK --- en el menú principal

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {

        // Copiamos la imagen normal
        img_click = img_normal.clone();
        // ---- BOTON CORAZÓN ----
        if (inside(btn_corazon, x, y)) {
            drawMask(img_click, btn_corazon);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);   // pequeña pausa visual
            encenderVentana(1);
        }

        // ---- BOTON HUESOS ----
        else if (inside(btn_huesos, x, y)) {
            drawMask(img_click, btn_huesos);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: HUESOS\n";
            encenderVentana(2);
        }

        // ---- BOTON PULMONES ----
        else if (inside(btn_pulmones, x, y)) {
            drawMask(img_click, btn_pulmones);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PULMONES\n";
            encenderVentana(3);
        }

        // ---- RESULTADOS ----
        else if (inside(btn_result, x, y)) {
            drawMask(img_click, btn_result);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: RESULTADOS\n";
            encenderVentana(4);
        }

        // ---- COMPARACION DnCNN ----
        else if (inside(btn_comp, x, y)) {
            drawMask(img_click, btn_comp);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: COMPARACION DnCNN\n";
            encenderVentana(5);
        }

        // ---- EXTRA ----
        else if (inside(btn_extra, x, y)) {
            drawMask(img_click, btn_extra);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: EXTRA\n";
            encenderVentana(6);
        }

        // ---- PRUEBAS ----
        
        else if (inside(btn_pruebas, x, y)) {
            drawMask(img_click, btn_pruebas);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PRUEBAS\n";
            encenderVentana(7);
        }

        // Restauramos la imagen normal después de la animación
        imshow("Aplicacion Principal", img_normal);
    }
}




Mat boneWindowing(Mat imagen, int minVal, int maxVal) {
    Mat salida = imagen.clone();
    double escala = 255.0 / (double)(maxVal - minVal);
    imagen.convertTo(salida, -1, escala, -minVal * escala);
    return salida;
}

// Mat defineBones(Mat imagen) {
//     Mat procesada;
//     procesada = boneWindowing(imagen, 120, 255);
//     GaussianBlur(imagen, procesada, cv::Size(3, 3), 0);
//     procesada = toClahe(procesada);

//     // 1. UMBRALIZACIÓN 
//     int t_strong = 170; 
//     int t_weak = 158; 

//     Mat fuerte, debil;
//     cv::threshold(procesada, fuerte, t_strong, 255, cv::THRESH_BINARY);
//     cv::threshold(procesada, debil, t_weak, 255, cv::THRESH_BINARY);

//     // -------------------------------------------------------------
//    //EROSIÓN 
//     // -------------------------------------------------------------
//     Mat kernelErosion = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
//     cv::erode(debil, debil, kernelErosion);
//     // -------------------------------------------------------------

//     Mat labels, stats, centroids;
//     int nLabels = cv::connectedComponentsWithStats(debil, labels, stats, centroids);

//     Mat mascaraBase = Mat::zeros(imagen.size(), CV_8UC1);

//     // Filtro de área (INTACTO)
//     int minArea = 25; 

//     // --- PARÁMETROS DE LA ZONA DE MUERTE ---
//     int zmX = 263;      
//     int zmY = 210;      
//     int zmRadio = 94;   
//     int areaColumnaGigante = 10000; 

//     // --- CONFIGURACIÓN DEL INTERVALO DE SLICES ---
//     // La zona de muerte solo se activa en estos slices
//     int inicioZonaMuerte = 245; 
//     int finZonaMuerte = 438;
//     // ---------------------------------------------

//     for(int i = 1; i < nLabels; i++) {
//         // Filtro básico de ruido
//         if (stats.at<int>(i, cv::CC_STAT_AREA) < minArea) continue;

//         Mat maskIsla = (labels == i);

//         // Solo aplicamos esto si estamos en los slices del estómago
//         if (currentSlice >= inicioZonaMuerte && currentSlice <= finZonaMuerte) {
            
//             double cX = centroids.at<double>(i, 0);
//             double cY = centroids.at<double>(i, 1);

//             // Distancia euclidiana al centro de la zona de muerte
//             double dist = std::sqrt(std::pow(cX - zmX, 2) + std::pow(cY - zmY, 2));

//             // Si el objeto está DENTRO del círculo de muerte
//             if (dist < zmRadio) {
//                 // Si NO es la columna gigante, lo matamos.
//                 if (stats.at<int>(i, cv::CC_STAT_AREA) < areaColumnaGigante) {
//                     continue; // Está en la zona prohibida y es pequeño -> BORRAR
//                 }
//             }
//         }

//         double minVal, maxVal;
//         cv::minMaxLoc(imagen, &minVal, &maxVal, NULL, NULL, maskIsla);

//         double umbralDureza = 180.0;

//         if (maxVal < umbralDureza) {
//             continue; // No tiene "corazón duro" -> BORRAR
//         }
//         // -------------------------------------------------------------

//         cv::bitwise_or(mascaraBase, maskIsla, mascaraBase);
//     }


//     // --- RELLENO Y RESTAURACIÓN (INTACTO) ---
//     Mat resultadoFinal = mascaraBase.clone();

//     // Dilatación para recuperar lo erosionado
//     cv::dilate(resultadoFinal, resultadoFinal, kernelErosion);

//     // Cierre
//     Mat kernelCierre = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
//     cv::morphologyEx(resultadoFinal, resultadoFinal, cv::MORPH_CLOSE, kernelCierre);

//     // Relleno final
//     vector<vector<cv::Point>> contours;
//     cv::findContours(resultadoFinal, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//     for(size_t i = 0; i < contours.size(); i++) {
//         cv::drawContours(resultadoFinal, contours, (int)i, Scalar(255), cv::FILLED);
//     }
    
//     return resultadoFinal;
// }
Mat defineBones(Mat imagen) {

    imagen = maskCircle2(imagen, 253, 264, 210, 147);

    Mat work = imagen.clone();
    if (work.type() != CV_8UC1) work.convertTo(work, CV_8UC1);

    // 1. MEJORA DE CONTRASTE (Siempre Sharpen)
    work = applySharpening(work);

    // 2. Blur (Kernel 3x3 fijo)
    GaussianBlur(work, work, cv::Size(3, 3), 0);

    // 3. Umbral (117 a 255 fijos)
    Mat mask = Umbrilize(work, 117, 255);

    // 4. Morfología básica 
    mask = close(mask, 3);
    
    // 5. Cerrar Gaps (Puenteo Agresivo)

    int kGap = 21; 
    Mat k = getStructuringElement(MORPH_ELLIPSE, cv::Size(kGap, kGap));
    dilate(mask, mask, k);
    mask = fillHoles(mask);
    erode(mask, mask, k);

    // 6. Filtro Dureza (Area/Media/ZonaMuerte)
    Mat maskFiltrada = filterByAreaAndIntensity(mask, work);

    // 7. Rellenar huecos finales
    maskFiltrada = fillHoles(maskFiltrada);

    return maskFiltrada;
}



Mat mergeAllMasks(
    Mat &imgOriginal,
    Mat *maskLung,   // verde
    Mat *maskHeart,  // rojo
    Mat *maskBone,   // amarillo
    float alpha = 0.3 //valor de transparencia
) 
{
    // Convertir a color si es necesario
    Mat imgColor;
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    Mat output = imgColor.clone();

    // Capas de color para cada máscara
    Scalar colorLung  = Scalar(0, 255, 0);   // verde
    Scalar colorHeart = Scalar(0, 0, 255);   // rojo
    Scalar colorBone  = Scalar(0, 255, 255); // amarillo

    int rows = imgColor.rows;
    int cols = imgColor.cols;

    // Recorrer píxeles una sola vez
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            Vec3b orig = imgColor.at<Vec3b>(i, j);
            Vec3b &dst = output.at<Vec3b>(i, j);

            // Copia inicial
            float r = orig[2], g = orig[1], b = orig[0];

            // LUNG
            if (maskLung && !maskLung->empty() && maskLung->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorLung[0];
                g = (1 - alpha) * g + alpha * colorLung[1];
                r = (1 - alpha) * r + alpha * colorLung[2];
            }

            // HEART
            if (maskHeart && !maskHeart->empty() && maskHeart->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorHeart[0];
                g = (1 - alpha) * g + alpha * colorHeart[1];
                r = (1 - alpha) * r + alpha * colorHeart[2];
            }

            // BONE
            if (maskBone && !maskBone->empty() && maskBone->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorBone[0];
                g = (1 - alpha) * g + alpha * colorBone[1];
                r = (1 - alpha) * r + alpha * colorBone[2];
            }

            dst = Vec3b((uchar)b, (uchar)g, (uchar)r);
        }
    }

    return output;
}
// MÉTODO CON LOS VALORES QUE OBTUVIMOS MEJORES PARA DEFINIR EL CORAZÓN, Y LOS PULMONES

Mat defineOrgan(Mat imagen, bool isLung) {

    int cxdf, cydf, radioXdf, radioYdf;
    int mindf, maxdf, kerneldf;
    int sliceLimit;

    double sigmadf = sigmaTrack / 10.0;
    
    if (isLung) {
        // Parámetros para PULMONES (Umbral bajo, Apertura, Máscara grande)
        cxdf = 253;
        cydf = 264;
        radioXdf = 210;
        radioYdf = 147;
        mindf = 0;
        maxdf = 58;
        kerneldf = 9;
        sliceLimit = 127; 
        
    } else {

        cxdf = 284;
        cydf = 220;
        radioXdf = 87;
        radioYdf = 65;
        mindf = 110;  // Umbral ajustado para tejido blando
        maxdf = 170;
        kerneldf = 3; 
        sliceLimit = 40; 
    }


    if (currentSlice < sliceLimit) {
        imagen = maskCircle(imagen, cxdf, cydf, radioXdf, radioYdf);
        GaussianBlur(imagen, imagen, cv::Size(kerneldf, kerneldf), sigmadf);

        // Umbralización
        imagen = Umbrilize(imagen, mindf, maxdf);
        
        // Operación Morfológica Condicional
        if (isLung) {
            imagen = open(imagen, kernelSize); // Pulmones usan Apertura

        } else {
            imagen = open(imagen, kernelSize); // Corazón usa Cierre
            imagen = close(imagen, kernelSize); // Corazón usa Cierre

            vector<vector<cv::Point>> contours;
            vector<Vec4i> hierarchy;
            // Encontrar contornos externos
            findContours(imagen, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            double maxArea = 0;
            int maxAreaIdx = -1;
            // Buscar el índice del contorno con el área más grande
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxAreaIdx = i;
                }
            }
            
            Mat cleanMask = Mat::zeros(imagen.size(), CV_8UC1);
            
            if (maxAreaIdx >= 0) {
                drawContours(cleanMask, contours, maxAreaIdx, Scalar(255), FILLED);
            }
            
            imagen = cleanMask;

        }

    } else {
        imagen = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
    }

    return imagen;
}

Mat sumarMascaras(Mat *maskLung, Mat *maskHeart, Mat *maskBone)
{
    // Crear máscara final vacía
    Mat finalMask = *maskLung;

    bitwise_or(finalMask, *maskHeart, finalMask);
    bitwise_or(finalMask, *maskBone, finalMask);

    return finalMask;
}




///INTERFACE METHODS, NOT IMPLEMENTED TILL THE END
void encenderCorazon(Mat original);
void encenderHueso(Mat original);
void encenderPulmones(Mat original);
void encenderResults(Mat original);
void encenderComparativa(Mat original);
void encenderExtra(Mat original);
void encenderPruebas(Mat original);

// MAIN

int main() {
    // menu principal
    img_normal = imread("img_normal.png");
    resize(img_normal,img_normal,cv::Size(500, 395));
    namedWindow("Aplicacion Principal", WINDOW_AUTOSIZE);
    setMouseCallback("Aplicacion Principal", onMouse);
    moveWindow("Aplicacion Principal", 0,0); // la posicionamos en 0,0 para ajustarlo despues
    
    
    string folder = "L333"; // carpeta de imagenes
    vector<string> files = getIMA(folder);   // lista de imagenes (.ima ) string

    imgs;

    imgs.reserve(files.size());// necesario para optimizar el uso de memoria


    for (auto& f : files) { // almacenamiento imagen a imagen de los archivos de la carpeta
        Mat m = readIMA(f);
        if (m.empty()) m = Mat::zeros(256, 256, CV_8UC1);
        imgs.push_back(m);
    }  
    
    // createTrackbar("Slice", "Aplicacion Principal", &currentSlice, imgs.size() - 1);

    // VENTANAS
    while (true) {
        
        // MOSTRAMOS EL MENÚ, SIEMPRE ES VISIBLE
        
        imshow("Aplicacion Principal", img_normal); //ESTA VENTANA YA TIENE LOS MENUS, ENTONCES, HACEMOS IF'S PARA CADA MENU
        Mat original = imgs[currentSlice].clone();

        if(corazon){// PROCESO COMPLETO PARA OBTENER MASCARA DE CORAZON
            encenderCorazon(original);

        } else if (hueso){ // PROCESO COMPLETO PARA OBTENER MASCARA DE HUESOS
            encenderHueso(original);

        } else if (pulmones){ // PROCESO COMPLETO PARA OBTENER MASCARA DE PULMONES
            encenderPulmones(original);
            
        } else if (result){ // RESULTADOS DE LA APLICACIÓN
            encenderResults(original);
            
        } else if (comparacion){ // COMPARACIÓN DE RESULTADOS Y BLUR GAUSSIANO CON LA RED DnCNN EN PYTHON
            encenderComparativa(original);
            
        } else if (extra){ // VALORES DE LAS IMAGENES AISLADAS Y RESALTADAS EN LAS ZONAS QUE NOS INTERESAN
            encenderExtra(original);
            
        } else if (pruebas){ // SLIDERS Y BOTONES PARA HACER PRUEBAS CON LAS IMAGENES
            encenderPruebas(original);
            
        }
        
        if (waitKey(30) == 27) break; // ESC para salir
    }

    destroyAllWindows();
    return 0;
}

// INTERFACES

void eliminarControles(){ 
    destroyWindow("Aplicacion Principal"); 
    namedWindow("Aplicacion Principal", WINDOW_AUTOSIZE); 
    setMouseCallback("Aplicacion Principal", onMouse); 
    createTrackbar("Slice", "Aplicacion Principal", &currentSlice, imgs.size() - 1); 
    moveWindow("Aplicacion Principal", 0,0); // la posicionamos en 0,0 para ajustarlo despues controles = false; 
}

void createSliceTrackbar() {
    int maxVal = imgs.empty() ? 0 : (int)imgs.size() - 1;
    if (currentSlice > maxVal) currentSlice = maxVal;
    if (currentSlice < 0) currentSlice = 0;

    createTrackbar("Slice", "Aplicacion Principal", &currentSlice, maxVal);
}
// Cerrar ventanas auxiliares
void cerrarVentanas() {
    static vector<string> windows = {
        "Original", "Region de interés", "Umbralización", "Apertura",
        "Sharpening", "Cierre", "CLAHE", "EQ", "Gauss", "Resultado", "Cierre2"
    };
    for (auto &w : windows) destroyWindow(w);
}


// Resetear o activar controles (caso Pruebas)
void enableControls() {
    controles = true;
    // Aquí pones los trackbars de pruebas
}


// Para interfaces normales (corazón, pulmones, hueso, etc.)
void prepareStandardView() {
    if (controles) eliminarControles();

    if (tiempo == 1) {
        cerrarVentanas();

        if (!slicer) {
            createSliceTrackbar();
            slicer = true;
        }

        tiempo = 2;
    }
}

void encenderCorazon(Mat original){

    prepareStandardView();

    int min = 110;
    int max = 170;
    int kernel = 3;
    double sigma = 1;

    int cxdf = 284;
    int cydf = 220;
    int radioXdf = 87;
    int radioYdf = 65;
    
    Mat roi;
    if(currentSlice <= 41){
        roi =  maskCircle(original, cxdf, cydf, radioXdf, radioYdf);
    } else {
        roi = Mat::ones(original.rows, original.cols, CV_8UC1);
    }

    Mat blur = toGaussianBlur(roi, kernel, sigma);
    
    Mat umbral = Umbrilize(blur, min, max);

    Mat apertura = open(umbral, kernel);

    Mat cierre = close(apertura, kernel);


    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    // Encontrar contornos externos
    findContours(cierre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int maxAreaIdx = -1;
    // Buscar el índice del contorno con el área más grande
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }
    
    Mat cleanMask = Mat::zeros(cierre.size(), CV_8UC1);
    
    if (maxAreaIdx >= 0) {
        drawContours(cleanMask, contours, maxAreaIdx, Scalar(255), FILLED);
    }
    
    Mat cierre2 = cleanMask;


    imshow("Original", original);
    moveWindow("Original", 500,0);
    imshow("Region de interés", roi);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*2,0);
    imshow("Apertura", apertura);
    moveWindow("Apertura", 500+(original.cols)*0,original.rows);
    imshow("Cierre", cierre);
    moveWindow("Cierre", 500+(original.cols)*1,original.rows);
    imshow("Cierre2", cierre2);
    moveWindow("Cierre2", 500+(original.cols)*2,original.rows);


}

void encenderHueso(Mat original){
    prepareStandardView();



    imshow("Original", original);
    moveWindow("Original", 500,0);


    Mat roi = maskCircle2(original, 253, 264, 210, 147);

    Mat sharp = applySharpening(roi);

    Mat Blur = toGaussianBlur(sharp, 3);
    
    Mat umbral = Umbrilize(original,117,255);

    Mat cierre = close(original,3);

    int kGap = 21; 
    Mat k = getStructuringElement(MORPH_ELLIPSE, cv::Size(kGap, kGap));
    Mat  flood;
    dilate(cierre, flood, k);
    flood = fillHoles(flood);
    erode(flood, flood, k);

    Mat maskFiltrada = filterByAreaAndIntensity(flood, original);

    Mat Cierre2 = fillHoles(maskFiltrada);



    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Region de interés", roi);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Sharpening", sharp);
    moveWindow("Sharpening", 500+(original.cols)*2,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*0,original.rows);
    imshow("Cierre", cierre);
    moveWindow("Cierre", 500+(original.cols)*1,original.rows);
    imshow("Mascara Filtrada", maskFiltrada);
    moveWindow("Mascara Filtrada", 500+(original.cols)*2,original.rows);
    imshow("Cierre2", Cierre2);
    moveWindow("Cierre2", 0,original.rows);
    

}

void encenderPulmones( Mat original){
    prepareStandardView();

    int cxdf, cydf, radioXdf, radioYdf;
    int mindf, maxdf, kerneldf;
    int sliceLimit;

    double sigmadf = 1;
    
    cxdf = 253;
    cydf = 264;
    radioXdf = 210;
    radioYdf = 147;
    mindf = 0;
    maxdf = 58;
    kerneldf = 9;
    sliceLimit = 127; 
    Mat roi;
    Mat blur;
    Mat umbral;
    Mat apertura;

    if(currentSlice <= 127){
        roi = maskCircle(original, cxdf, cydf, radioXdf, radioYdf);
        blur = toGaussianBlur(roi,kerneldf,sigmadf);
        umbral = Umbrilize(blur, mindf, maxdf);
        apertura = open(umbral, kernelSize);
    } else {
        roi = Mat::zeros(original.rows, original.cols,CV_8UC1);
        blur = roi;
        umbral = roi;
        umbral = roi;
        apertura = roi;
    }

    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Region de interés", roi);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Blur Gaussiano", blur);
    moveWindow("Blur Gaussiano", 500+(original.cols)*2,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*0,original.rows);
    imshow("Apertura", apertura);
    moveWindow("Apertura", 500+(original.cols)*1,original.rows);

}
void encenderResults(Mat original){
    prepareStandardView();

    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    Mat original1 = original.clone();
    Mat original2 = original.clone();
    Mat original3 = original.clone();
    Mat corazon = defineOrgan(original1,false);
    Mat pulmones = defineOrgan(original2,true);
    Mat huesos = defineBones(original3);
    
    Mat pulmones2 = pulmones.clone();
    Mat merged = mergeAllMasks(original, &pulmones2, &corazon, &huesos, 0.3);
    Mat binaria = sumarMascaras(&pulmones2, &corazon, &huesos);
    Mat isolated;
    bitwise_and(original, binaria, isolated);

    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Corazon", corazon);
    moveWindow("Corazon", 500+(original.cols)*1,0);
    imshow("Pulmones", pulmones);
    moveWindow("Pulmones", 500+(original.cols)*2,0);
    imshow("Huesos", huesos);
    moveWindow("Huesos", 500+(original.cols)*0,original.rows);
    imshow("Mezcla", merged);
    moveWindow("Merged", 500+(original.cols)*1,original.rows);
    imshow("Aislada", isolated);
    moveWindow("Aislada", 500+(original.cols)*2,original.rows);
    
    
    
}
void encenderComparativa(Mat original){
    prepareStandardView();

    ///  TODO: Quitar controles
    ///  TODO: resultado aislado
    ///  TODO: resultado obtenido con DnCNN // esto debe tardar un poco, agregamos un wait
    
}
void encenderExtra(Mat original){
    prepareStandardView();

    ///  TODO: Quitar controles
    ///  TODO: imagen con los organos aislados
    ///  TODO: imagenes con los pulmones resaltados, dado la opinion del radiologo
    ///  TODO: imagen merge, de los organos realzados
    
}


void encenderPruebas(Mat original){
    controles = true;
    ///  TODO: Colocar controles
    ///  TODO: imagen original
    ///  TODO: imagen filtro CLAHE
    ///  TODO: imagen filtro EQ
    ///  TODO: imagen filtro Sharpening
    ///  TODO: imagen filtro Gaussiano
    ///  TODO: imagen filtro Umbralizacion
    ///  TODO: imagen filtro close
    ///  TODO: imagen filtro open
    
}
