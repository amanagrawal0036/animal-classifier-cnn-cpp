#include <iostream>
#include <string>

#include "image_loader.h"
#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>

#include <exception>
#include <vector>

#include <utility>
#include <algorithm>

using namespace std;

//This fn loads image from filepath, converts it to grayscale image, converts it to 2d array and then normalizes the values from 0-1
ImageData loadImage(const string& relativeFilepath, int label) {
    ImageData result;
    result.label = label; // Assign the label regardless of load success
    string fullPath = "images/" + relativeFilepath;
    // Load the image in grayscale using the constructed full path
    cv::Mat img = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);

    // Check if the image loading failed
    if (img.empty()) {
        cerr << "Error loading image: " << fullPath << '\n';
        return result;
    }

    if (img.size() != cv::Size(150, 150)) {
        cv::resize(img, img, cv::Size(150, 150), cv::INTER_LINEAR);
    }

   // Converting Matrice to 2d vector<vector<float>> and normalizing it 
    result.data.resize(img.rows); // Resize outer vector to number of rows
    for (int y = 0; y < img.rows; ++y) {
        result.data[y].resize(img.cols);
        for (int x = 0; x < img.cols; ++x) {
             //cout << img.at(x,y) <<"\t";

            //dividing it by 255 because grayscale images are in the range of 0-255 so for normalizing
            result.data[y][x] = img.at<unsigned char>(y, x) / 255.0f;
        }
    }

    return result;
}
//function that reads image location and its corresponding label from a CSV file
vector<pair<string, int>> readCSV(const string& filename) {
    vector<pair<string, int>> data; 
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening CSV file: " << filename << '\n';
        return data;
    }

    string line;
    if (!getline(file, line)) {
         cerr << "Error reading header line or file is empty: " << filename << '\n';
         return data;
    }
    while (getline(file, line)) {
    
        stringstream ss(line);
        string path_part;
        string label_part; 

        // Try to extract the path and label parts separated by a comma
        if (getline(ss, path_part, ',') && getline(ss, label_part)) {
        
            size_t last_char_pos = label_part.find_last_not_of(" \n\r\t");
            if (string::npos != last_char_pos) {

                label_part.erase(last_char_pos + 1);
            } else {
                 cerr << "Warning: Found empty or whitespace-only label for path '" << path_part << "' in line: \"" << line << "\". Skipping.\n";
                 continue;
            }
            data.emplace_back(path_part, stoi(label_part));


        } else if (!line.empty() && !path_part.empty()) {

            cerr << "err reading line, missing comma or label?:) " << line << '\n';
        }
    }
    file.close();
    return data;
}

