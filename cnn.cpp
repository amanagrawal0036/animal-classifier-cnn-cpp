#include <iostream>

#include "cnn.h"
#include "image_loader.h"

#include <vector>
#include <cmath>

#include <numeric>
#include <algorithm>
#include <random>

#include <iomanip>
using namespace std;

// Helper to convert ImageData a tensor
Tensor3D CNN::imageToInputTensor(const ImageData& img) {
    size_t height = 0;
    size_t width = 0;
    if (!img.data.empty()) {
        height = img.data.size();
        if (!img.data[0].empty()) {
            width = img.data[0].size();
        }
    }
    //cout<<height <<"  "<<width<<endl;
    // using one dimenstion as 1 , giving it a const value because we are grayscaling our dataset
    Tensor3D tensor = createTensor3D(1, height, width);
    if(height > 0) {
        tensor[0] = img.data;
    }
    return tensor;
}

// Converting label to one-hot vector
Tensor1D CNN::toOneHot(int label, int num_classes) {
    Tensor1D one_hot(num_classes, 0.0f);
    if (label >= 0 && label < num_classes) {
        one_hot[label] = 1.0f;
    }
    return one_hot;
}

// Forward pass through all layers
Tensor3D CNN::forward(const Tensor3D& input) {
    Tensor3D current_output = input;
    for (const auto& layer : layers) {
        current_output = layer->forward(current_output);
    }
    return current_output;
}

// Backward pass through all layers
void CNN::backward(const Tensor3D& gradient, float learning_rate) {
    Tensor3D current_gradient = gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        current_gradient = (*it)->backward(current_gradient, learning_rate);
    }
}

// Calculate Cross-Entropy Loss
float CNN::calculateLoss(const Tensor1D& predicted_probs, int true_label) {
    // Add small epsilon to prevent log(0)
    float prob = 1e-9f;
    if (true_label >= 0 && static_cast<size_t>(true_label) < predicted_probs.size()) {
         prob = max(predicted_probs[true_label], 1e-9f);
    }
    return -log(prob);
}

// Train the network
void CNN::train(const vector<ImageData>& training_data,
                  int num_classes,
                  int epochs,
                  float learning_rate,
                  int batch_size)
{
    if (layers.empty() || training_data.empty()) {
        cout << "warning: can't train with empt n/w or data." << endl;
        return;
    }

    softmax_layer_ptr = dynamic_cast<SoftmaxLayer*>(layers.back().get());
    if (!softmax_layer_ptr) {
         cout << "warning: Last layer is not a SoftmaxLayer, training might proceed incorrectly." << endl;
    }


    size_t n_samples = training_data.size();
    vector<int> indices(n_samples);
    iota(indices.begin(), indices.end(), 0);

    cout << "\n--- Starting Training ---" << endl;
    cout << "Epochs: " << epochs << ", Learning Rate: " << learning_rate
         << ", Samples: " << n_samples << ", SGD (Batch Size=" << batch_size << ")" << endl;
    cout << fixed << setprecision(5);

    //Epoch
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data indices for each epoch (important for SGD)
        shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

        float total_epoch_loss = 0.0f;
        int correct_predictions = 0;

        //Training Loop (iterating through shuffled datapoints)
        for (int i = 0; i < n_samples; ++i) {
            const ImageData& sample = training_data[indices[i]];
             if (sample.data.empty()) {

                 continue;
             }

             int true_label = sample.label;
             if (true_label < 0 || true_label >= num_classes) {

                 continue;
             }
            
             // 1. Convert image to Tensor3D input
            Tensor3D input_tensor = imageToInputTensor(sample);
            //cout<<"debug 123"<<endl;
            // 2. Forward Pass
            Tensor3D output_tensor = this->forward(input_tensor);

            // fetch 1D prob from the output tensor
            Tensor1D predicted_probs(num_classes);
             if (getDepth(output_tensor) == static_cast<size_t>(num_classes) && getHeight(output_tensor) == 1 && getWidth(output_tensor) == 1) {
                 for(int k=0; k<num_classes; ++k) predicted_probs[k] = output_tensor[k][0][0];
             } else {

                  cout << "Warning: Output tensor shape mismatch for sample " << indices[i] << endl;
                  continue;
             }
            
             // calculate Loss
            float loss = calculateLoss(predicted_probs, true_label);
            total_epoch_loss += loss;

            // calculate softmax output - OneHot label
            Tensor1D one_hot_label = toOneHot(true_label, num_classes);
            Tensor3D initial_gradient;
            if (softmax_layer_ptr) {
                 initial_gradient = softmax_layer_ptr->calculateInitialGradient(one_hot_label);
            } else {
                 initial_gradient = createTensor3D(num_classes, 1, 1, 0.0f);
                 cout << "Warning: we create zero gradient as last layer is not softmax." << endl;
            }

            // Update Weights)
            this->backward(initial_gradient, learning_rate);

            // count correct preds
            int predicted_label = distance(predicted_probs.begin(), max_element(predicted_probs.begin(), predicted_probs.end()));
            if (predicted_label == true_label) {
                correct_predictions++;
            }
            //Progress Update pretty print:))
             if ((i + 1) % 100 == 0 || (i + 1) == n_samples) {
                 cout << "\rEpoch [" << epoch + 1 << "/" << epochs << "] Sample [" << i + 1 << "/" << n_samples << "] Avg Loss: " << (total_epoch_loss / (i + 1)) << flush;
             }

        }

        float epoch_avg_loss = (n_samples > 0) ? total_epoch_loss / n_samples : 0.0f;
        float epoch_accuracy = (n_samples > 0) ? static_cast<float>(correct_predictions) / n_samples : 0.0f;

        cout << "\rEpoch [" << epoch + 1 << "/" << epochs << "] Completed. "
             << "Avg Loss: " << epoch_avg_loss << ", Accuracy: " << epoch_accuracy * 100.0f << "%" << endl;

    }
     cout << "--- Training Finished ---" << endl;
}

// Predict the class for one image
int CNN::predict(const ImageData& image) {
     if (layers.empty() || image.data.empty()) {

         return -1;
     }

     Tensor3D input_tensor = imageToInputTensor(image);
     Tensor3D output_tensor = this->forward(input_tensor);

     size_t num_classes = getDepth(output_tensor);
      if (num_classes == 0 || getHeight(output_tensor)!=1 || getWidth(output_tensor)!=1) {

           return -1;
      }
      cout<<getHeight(output_tensor)<<" "<<getWidth(output_tensor)<<endl;
      
     Tensor1D predicted_probs(num_classes);
     for(int k=0; k<num_classes; ++k) {
         predicted_probs[k] = output_tensor[k][0][0];
     }

    int predicted_label = distance(predicted_probs.begin(),
                                     max_element(predicted_probs.begin(), predicted_probs.end()));
    return predicted_label;
}