#ifndef CNN_H
#define CNN_H

#include <iostream>
#include <string>

#include "cnn_layers.h"
#include "image_loader.h"

#include <vector>
#include <memory>
#include <tuple>
#include <utility>

using namespace std;

class CNN {
public:
    CNN() = default;
    ~CNN() = default; // unique_ptr handles layer cleanup

    // Add a layer to the network
    template<typename LayerType, typename... Args>
    void addLayer(Args&&... args) {
        // Get the input shape requirement from the previous layer, if any
        size_t input_d = 1, input_h = 0, input_w = 0;
        if (!layers.empty()) {
            tie(input_d, input_h, input_w) = layers.back()->getOutputShape();
        } else {
        }
        // Construct the layer in place using perfect forwarding
        layers.emplace_back(make_unique<LayerType>(std::forward<Args>(args)...));
         cout << "Added Layer. Network size: " << layers.size() << endl;
         auto [od, oh, ow] = layers.back()->getOutputShape();
         cout << " -> Output Shape: (" << od << ", " << oh << ", " << ow << ")" << endl;
    }

    // training network
    void train(const vector<ImageData>& training_data,
               int num_classes,
               int epochs,
               float learning_rate,
               int batch_size = 1); // sigmoid
    
    // Predict the class for a single image
    int predict(const ImageData& image);
    // Calculate Cross-Entropy Loss for a single prediction
    float calculateLoss(const Tensor1D& predicted_probs, int true_label);


private:
    vector<unique_ptr<Layer>> layers;
    SoftmaxLayer* softmax_layer_ptr = nullptr;
    // converts label to one-hot encoding
    Tensor1D toOneHot(int label, int num_classes);
    // fwd pass through all layers
    Tensor3D forward(const Tensor3D& input);
    // backward pass through all layers
    void backward(const Tensor3D& gradient, float learning_rate);
    // Converts ImageData to Tensor3D (grayscale)
    Tensor3D imageToInputTensor(const ImageData& img);
};

#endif