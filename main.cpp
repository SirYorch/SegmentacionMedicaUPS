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


using InputPixelType  = short;
using OutputPixelType = unsigned char;
using InputImageType  = Image<InputPixelType, 2>;
using OutputImageType = Image<OutputPixelType, 2>;

int currentSlice = 0;
int ksizeTrack = 3;       // representará kernel = 2*ksizeTrack + 1
int sigmaTrack = 10;      // representará sigma = sigmaTrack / 10.0
int umbralMin = 0;
int umbralMax = 255;
int cx = 100;
int cy = 100;
int radioX = 50;
int radioY = 50;

int kernelSize = 3;

bool clahe = false;
bool eq = false;


void updateKSize(int KSize){
    KSize = KSize*2+1;
    
}

//en este método se aplican los umbrales de mayor y menor para poder definir los mejores valores
Mat Umbrilize(Mat imagen, int umbralMin, int umbralMax){
    Mat img = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
    
    for(int i =0 ; i < imagen.rows ; i ++){
        for(int j =0 ; j < imagen.cols ; j ++){
            uchar pixel = imagen.at<uchar>(i, j);

            if (pixel >= umbralMin && pixel <= umbralMax)
                img.at<uchar>(i, j) = 255;     // Blanco (dentro del rango)
            else
                img.at<uchar>(i, j) = 0;       // Negro (fuera del rango)      
        }   
    }
    return img;
}

// Convertir ITK → OpenCV
Mat ITKToMat(OutputImageType::Pointer img) {
    auto size = img->GetLargestPossibleRegion().GetSize();
    Mat out(size[1], size[0], CV_8UC1);
    memcpy(out.data, img->GetBufferPointer(), size[0] * size[1]);
    return out;
}

// Leer como RAW (fallback)
Mat readRaw(const string& filename) {
    ifstream f(filename, ios::binary);
    if (!f.is_open()) return Mat();

    f.seekg(0, ios::end);
    size_t total = f.tellg();
    f.seekg(0, ios::beg);

    size_t pixels = total / 2;
    int side = sqrt(pixels);

    vector<short> buffer(side * side);
    f.read((char*)buffer.data(), total);

    Mat img16(side, side, CV_16SC1, buffer.data());
    Mat img8;
    normalize(img16, img8, 0, 255, NORM_MINMAX);
    img8.convertTo(img8, CV_8UC1);

    return img8.clone();
}

// Leer IMA con ITK o RAW
Mat readIMA(const string& filename) {
    try {
        using ImageIOType = GDCMImageIO;
        ImageIOType::Pointer dicomIO = ImageIOType::New();

        using ReaderType = ImageFileReader<InputImageType>;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetImageIO(dicomIO);
        reader->SetFileName(filename);
        reader->Update();

        using RescaleType = RescaleIntensityImageFilter<InputImageType, OutputImageType>;
        RescaleType::Pointer scale = RescaleType::New();
        scale->SetInput(reader->GetOutput());
        scale->SetOutputMinimum(0);
        scale->SetOutputMaximum(255);
        scale->Update();

        return ITKToMat(scale->GetOutput());
    }
    catch (...) {
        cout << "Error leyendo como IMA/DICOM: " << filename << "\n";
        return Mat();
    }
}

Mat toClahe(Mat imagen){
    Mat salida;

    Ptr<CLAHE> clahe = createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(imagen, salida);

    return salida;
}


Mat toEq(Mat imagen){
    
    equalizeHist(imagen, imagen);
    
    return imagen;
}

Mat toGaussianBlur(Mat imagen, int kernelSize = 5, double sigma = 0) {
    Mat salida;

    // El kernel debe ser impar
    if (kernelSize % 2 == 0) kernelSize++;

    GaussianBlur(imagen, salida, cv::Size(kernelSize, kernelSize), sigma);

    return salida;
}


// Obtener lista de .ima
vector<string> getIMA(const string& dir) {
    vector<string> files;

    for (auto& e : filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;

        string ext = e.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".ima")   // ahora acepta .IMA y .ima
            // cout << e << endl;  
            files.push_back(e.path().string());
    }

    sort(files.begin(), files.end());
    return files;
}

void activateClahe(int estado, void*) {
    if (estado)
        clahe << true;
    else
        clahe << false;
}


void activateEq(int estado, void*) {
    if (estado)
        eq << true;
    else
        eq << false;
}




// Declaración global de botones
Rect btnCLAHE(10, 10, 120, 40);
Rect btnEq(10, 60, 120, 40);

// Estado de botones (si quieres alternar)


Mat maskCircle(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);

    // 1. Dibujar círculo blanco = zona visible
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);

    // 2. Crear salida blanca (color blanco completo)
    Mat salida(img.rows, img.cols, img.type(), Scalar(255));

    // 3. Copiar dentro del círculo la imagen original
    img.copyTo(salida, mask);

    return salida;
}


void onMouse(int event, int x, int y, int flags, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        if (btnCLAHE.contains(cv::Point(x, y))) {
            clahe = !clahe;
            cout << "CLAHE: " << (clahe ? "ON\n" : "OFF\n");
        }
        if (btnEq.contains(cv::Point(x, y))) {
            eq = !eq;
            cout << "Eq: " << (eq ? "ON\n" : "OFF\n");
        }
        
    }

    // Redibujar controles para feedback visual (opcional pero recomendado)
        Mat panel(60, 300, CV_8UC3, Scalar(50, 50, 50));
        
        // // Color del botón según estado
        // Scalar colorClahe = clahe ? Scalar(100, 255, 100) : Scalar(200, 200, 200);
        // Scalar colorEq = eq ? Scalar(100, 255, 100) : Scalar(200, 200, 200);

        // rectangle(panel, btnCLAHE, colorClahe, FILLED);
        // putText(panel, "CLAHE", cv::Point(25, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));

        // rectangle(panel, btnEq, colorEq, FILLED);
        // putText(panel, "EQ HIST", cv::Point(35, 85), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));

        imshow("Controles", panel);

}
Mat open(Mat imagen, int kernelSize){
    // Retorna proceso de apertura, erosion y dilatacion
    Mat erodida, salida;

    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));

    erode(imagen, erodida, kernel);
    dilate(erodida, salida, kernel);
    return salida;
}

Mat close(Mat imagen, int kernelSize){
    // Retorna proceso de apertura, erosion y dilatacion
    Mat dilatada, salida;

    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));

    dilate(imagen, dilatada, kernel);
    erode(dilatada, salida, kernel);
    
    return salida;
}

Mat defineBones(Mat imagen) {

    int minBone = 150; 
    int maxBone = 255;
    int kernelBone = 1; // Kernel pequeño para no fusionar costillas cercanas

    // 2. Umbralización (Lo más importante)
    imagen = Umbrilize(imagen, minBone, maxBone);
    imagen = close(imagen, kernelBone);
    imagen = open(imagen, kernelBone);

    return imagen;
}

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
            
            // Crear una máscara negra limpia
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


Mat mergeMasks(Mat &imgOriginal, Mat &maskLung, float alpha = 0.3) {
    Mat imgColor;

    // Convertir a BGR si la original es gris
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Crear capa verde del mismo tamaño
    Mat greenLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 255, 0));

    // Crear salida inicial como copia
    Mat output = imgColor.clone();

    // Aplicar overlay SOLO donde la máscara es 255
    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {

            if (maskLung.at<uchar>(i, j) == 255) {

                // output = (1 - alpha) * original + alpha * green
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * greenLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * greenLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * greenLayer.at<Vec3b>(i, j)[2];
            }
        }
    }

    return output;
}


Mat mergeMasksHeart(Mat &imgOriginal, Mat &maskLung, float alpha = 0.3) {
    Mat imgColor;

    // Convertir a BGR si la original es gris
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Crear capa verde del mismo tamaño
    Mat greenLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 0, 255));

    // Crear salida inicial como copia
    Mat output = imgColor.clone();

    // Aplicar overlay SOLO donde la máscara es 255
    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {

            if (maskLung.at<uchar>(i, j) == 255) {

                // output = (1 - alpha) * original + alpha * green
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * greenLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * greenLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * greenLayer.at<Vec3b>(i, j)[2];
            }
        }
    }

    return output;
}

Mat mergeMasksBones(Mat &imgOriginal, Mat &maskBone, float alpha = 0.3) {
    Mat imgColor;

    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Color AMARILLO: (Blue=0, Green=255, Red=255)
    Mat yellowLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 255, 255));

    Mat output = imgColor.clone();

    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {
            if (maskBone.at<uchar>(i, j) == 255) {
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[2];
            }
        }
    }
    return output;
}




int main() {
    string folder = "L333";
    auto files = getIMA(folder);

    vector<Mat> imgs;
    imgs.reserve(files.size());

    for (auto& f : files) {
        Mat m = readIMA(f);
        if (m.empty()) m = Mat::zeros(256, 256, CV_8UC1);
        imgs.push_back(m);
    }

    if (imgs.empty()) {
        cout << "No se pudo cargar ninguna imagen\n";
        return 1;
    }
    namedWindow("Controles", WINDOW_AUTOSIZE);


    createTrackbar("Slice",    "Controles", &currentSlice, imgs.size() - 1);
    //createTrackbar("KSize",    "Controles", &ksizeTrack, 50);  // límite arbitrario
    
    Mat panel(60, 300, CV_8UC3, Scalar(50, 50, 50));

    ///// BOTONES  NOS SIRVEN PARA DARLE ACCIONES A LA APLICACIÓN DESDE OTRA VENTANA

    setMouseCallback("Controles", onMouse);


    // Con pruebas llegamos a la idea de que el BLUR aplicado solo sirve para los pulmones
    // Los huesos se ven mucho mejor con la ecualización y luego BLUR, 
    // Y el corazon se ve mejor con 

    // Los pulmones se ven mucho mejor solo con apertura

    /// para aislar los pulmonesel circulo usamos. slices 0-127
    // circulo en 253,264
    // RadioX de 210
    // radioY de 247

    // El corazon se ven mucho mejor solo con cierre

    /// para aislar los pulmonesel circulo usamos. slices 0-42
    // circulo en 284,215
    // RadioX de 89
    // radioY de 66


    imshow("Controles", panel);
    moveWindow("Controles", 0, 0);


    namedWindow("Pulmones", WINDOW_AUTOSIZE);
    namedWindow("CompletaPulmones", WINDOW_AUTOSIZE);
    namedWindow("CorazonMascara", WINDOW_AUTOSIZE);
    namedWindow("CompletaCorazon", WINDOW_AUTOSIZE); 
    namedWindow("HuesosMascara", WINDOW_AUTOSIZE);
    namedWindow("CompletaHuesos", WINDOW_AUTOSIZE); 


    moveWindow("Pulmones", 300, 500);
    moveWindow("CompletaPulmones", 300, 0);
    moveWindow("CorazonMascara", 810, 500); 
    moveWindow("CompletaCorazon", 810, 0); 
    moveWindow("HuesosMascara", 1320, 500); 
    moveWindow("CompletaHuesos", 1320, 0); 


    while (true) {
        Mat Original = imgs[currentSlice].clone();

        // --- Procesamiento de Pulmones (como ya lo tenías) ---
        Mat pulmonesMascara = imgs[currentSlice].clone();
        pulmonesMascara = defineOrgan(pulmonesMascara, true); // Función que usa Apertura
        Mat completaPulmones = mergeMasks(Original, pulmonesMascara, 0.3); // Overlay verde

        imshow("Pulmones", pulmonesMascara);
        imshow("CompletaPulmones", completaPulmones);


        Mat corazonMascara = imgs[currentSlice].clone();
        corazonMascara = defineOrgan(corazonMascara, false);
        Mat completaCorazon = mergeMasksHeart(Original, corazonMascara, 0.3); // Overlay rojo

        imshow("CorazonMascara", corazonMascara); // Muestra solo la máscara del corazón
        imshow("CompletaCorazon", completaCorazon); // Muestra la imagen original con el corazón resaltado

        Mat huesosMascara = imgs[currentSlice].clone();
        huesosMascara = defineBones(huesosMascara);
        Mat completaHuesos = mergeMasksBones(Original, huesosMascara, 0.4); // Alpha un poco más alto para que brillen

        imshow("HuesosMascara", huesosMascara);
        imshow("CompletaHuesos", completaHuesos);

        int k = waitKey(30);
        if (k == 27) break; 
    }

    destroyAllWindows(); // Cierra todas las ventanas al salir
    return 0;
}