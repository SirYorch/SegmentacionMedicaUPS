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

    namedWindow("Visor", WINDOW_AUTOSIZE);
    namedWindow("VisorFiltro", WINDOW_AUTOSIZE);
    namedWindow("Blured", WINDOW_AUTOSIZE);
    namedWindow("Apertura", WINDOW_AUTOSIZE);
    namedWindow("Close", WINDOW_AUTOSIZE);
    namedWindow("ap+cl", WINDOW_AUTOSIZE);
    namedWindow("cl+ap", WINDOW_AUTOSIZE);
    

    moveWindow("Visor", 300, 0);        // Mover la ventana "Visor"
    moveWindow("Controles", 0, 0);
    moveWindow("VisorFiltro",810, 0);
    moveWindow("Blured", 1320, 0);
    moveWindow("Apertura", 0, 800);
    moveWindow("Close", 510, 800);
    moveWindow("ap+cl", 1020, 800);
    moveWindow("cl+ap", 1530, 800);

    createTrackbar("Slice",    "Controles", &currentSlice, imgs.size() - 1);
    createTrackbar("KSize",    "Controles", &ksizeTrack, 50);  // límite arbitrario
    createTrackbar("Sigma x10","Controles", &sigmaTrack, 100); // sigma max = 10.0

    createTrackbar("KernelSize","Controles", &kernelSize, 100); // sigma max = 10.0

    createTrackbar("Min", "Controles", &umbralMin, 255);  // límite arbitrari
    createTrackbar("Max", "Controles", &umbralMax, 255); // sigma max = 10.0
    
    Mat panel(160, 300, CV_8UC3, Scalar(50, 50, 50));



    ///// BOTONES  NOS SIRVEN PARA DARLE ACCIONES A LA APLICACIÓN DESDE OTRA VENTANA

    setMouseCallback("Controles", onMouse);

    // Botón CLAHE
    rectangle(panel, btnCLAHE, Scalar(200, 200, 200), FILLED);
    putText(panel, "CLAHE", cv::Point(25, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));

    // Botón BLUR
    rectangle(panel, btnEq, Scalar(200, 200, 200), FILLED);
    putText(panel, "BLUR", cv::Point(35, 85), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));

    imshow("Controles", panel);
    
    


   while (true) {
        Mat imagen = imgs[currentSlice].clone();
        Mat imagenOriginal = imgs[currentSlice].clone();

        int kernel = 2 * ksizeTrack + 1;
        double sigma = sigmaTrack / 10.0;
        
        if(eq){
            imagen = toEq(imagen);
        }

        if(clahe){
            imagen = toClahe(imagen);
        }
        
        GaussianBlur(imagen, imagen, cv::Size(kernel, kernel), sigma);
        
        Mat imagenBlur = imagen.clone();        

        // Umbralice
        // cout << umbralMin << " " << umbralMax;
        Mat salidaUmbral = Umbrilize(imagen, umbralMin , umbralMax);

        Mat Apertura = open(salidaUmbral, kernelSize);

        Mat Close = close(salidaUmbral, kernelSize);

        Mat apcl = close(Apertura, kernelSize);
        Mat clap = open(Close, kernelSize);


        // Mostrar
        imshow("Visor", salidaUmbral);
        imshow("VisorFiltro", imagen);
        imshow("Blured", imagenBlur);
        imshow("Apertura", Apertura);
        imshow("ap+cl", apcl);
        imshow("cl+ap", clap);
        imshow("Close", Close);

        int k = waitKey(30);
        if (k == 27) break; 
    }



    // Con pruebas llegamos a la idea de que el BLUR aplicado solo sirve para los pulmones
    // Los huesos se ven mucho mejor con la ecualización y luego BLUR, 



    // Los pulmones se ven mucho mejor solo con apertura


    return 0;
}
