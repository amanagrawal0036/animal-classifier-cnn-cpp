#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include "tensor_utils.h"
#include <vector>
#include <memory>

#include <tuple> 

using namespace std;

// This file contains def of all types of CNN layers
// Base class for all layers

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor3D forward(const Tensor3D& input) = 0;
    virtual Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) = 0;

    virtual tuple<size_t, size_t, size_t> getOutputShape() const = 0;

protected:
    Tensor3D last_input; 
    Tensor3D last_output; 
    tuple<size_t, size_t, size_t> input_shape;
    tuple<size_t, size_t, size_t> output_shape;
};

//Convolutional Layer inherits from layer
class ConvLayer : public Layer {
public:
    ConvLayer(size_t input_depth, size_t input_height, size_t input_width,
              size_t num_filters, size_t kernel_size, size_t stride = 1); 

    Tensor3D forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override;
    tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }

private:
    size_t num_filters;
    size_t kernel_size;
    size_t stride;
    
    vector<Tensor3D> filters;
    
    Tensor1D biases;

    
    float convolveSingleStep(const Tensor3D& input_slice, const Tensor3D& filter_kernel, float bias);
};

// ReLU activation Layer- simple max(0, x)
class ReLULayer : public Layer {
public:
     ReLULayer(size_t input_depth, size_t input_height, size_t input_width);
     Tensor3D forward(const Tensor3D& input) override;
     Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override;
     tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }
};

//Max pooling layer
class MaxPoolingLayer : public Layer {
public:
    MaxPoolingLayer(size_t input_depth, size_t input_height, size_t input_width,
                    size_t pool_size, size_t stride);

    Tensor3D forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override;
    tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }

private:
    size_t pool_size;
    size_t stride;
    
    
    vector<vector<vector<pair<size_t, size_t>>>> max_indices;
};

// Flatten - takes a 3D tensor and makes it 1D (well, 1x1xN)
class FlattenLayer : public Layer {
public:
    FlattenLayer(size_t input_depth, size_t input_height, size_t input_width);

    Tensor3D forward(const Tensor3D& input) override;
    Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override; 
    tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }
};


//Fully Connected Layer
class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size); 

    Tensor3D forward(const Tensor3D& input) override; 
    Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override; 
    tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }

private:
    size_t input_size;
    size_t output_size;
    
    Tensor2D weights;
    
    Tensor1D biases;
};


// Softmax - converts outputs to probabilities, usually the last layer for classification
class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(size_t input_size); 

    Tensor3D forward(const Tensor3D& input) override; 
    Tensor3D backward(const Tensor3D& output_gradient, float learning_rate) override;     
    Tensor3D calculateInitialGradient(const Tensor1D& true_label_one_hot);

    tuple<size_t, size_t, size_t> getOutputShape() const override { return output_shape; }
private:
     size_t input_size;
};


#endif
