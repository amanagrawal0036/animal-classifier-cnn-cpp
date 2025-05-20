#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>

#include <vector>
#include <utility>

using namespace std;

struct ImageData {
    vector<vector<float>> data;
    int label;

    ImageData() : label(-1) {}

    ImageData(ImageData&& other) noexcept
        : data(move(other.data)), label(other.label) {}

    ImageData& operator=(ImageData&& other) noexcept {
        if (this != &other) {
            data = move(other.data);
            label = other.label;
        }
        return *this;
    }

    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;
};

ImageData loadImage(const string& filepath, int label);
vector<pair<string, int>> readCSV(const string& filename);

#endif