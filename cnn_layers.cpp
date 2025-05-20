#include "cnn_layers.h"
#include <vector>
#include <cmath>
#include <limits> 
#include <algorithm> 
#include <numeric> 
#include <iostream> 

using namespace std; 

//below are the implementation of all the layers
ConvLayer::ConvLayer(size_t input_depth, size_t input_height, size_t input_width,
                     size_t num_filters, size_t kernel_size, size_t stride)
    : num_filters(num_filters), kernel_size(kernel_size), stride(stride)
{
    input_shape = {input_depth, input_height, input_width};
    // Calculate output dimensions (assuming no padding)
    size_t output_height = (input_height - kernel_size) / stride + 1;
    size_t output_width = (input_width - kernel_size) / stride + 1;
    output_shape = {num_filters, output_height, output_width};

    filters.resize(num_filters);
    for(int f = 0; f < num_filters; ++f) {
        // Each filter has depth matching input depth
        filters[f] = createTensor3D(input_depth, kernel_size, kernel_size);
        initializeTensor3D(filters[f], -0.1f, 0.1f); 
    }
    biases = Tensor1D(num_filters, 0.0f); 
}

// Simple helper fn for a single convolution step (dot product + bias)
float ConvLayer::convolveSingleStep(const Tensor3D& input_slice, const Tensor3D& filter_kernel, float bias) {
    float activation = 0.0f;
    for (int d = 0; d < getDepth(input_slice); ++d) {
        for (int r = 0; r < getHeight(input_slice); ++r) {
            for (int c = 0; c < getWidth(input_slice); ++c) {
                activation += input_slice[d][r][c] * filter_kernel[d][r][c];
            }
        }
    }
    return activation + bias;
}


Tensor3D ConvLayer::forward(const Tensor3D& input) {
    last_input = input; 
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D output = createTensor3D(out_depth, out_height, out_width);

    for (int f = 0; f < num_filters; ++f) { 
        for (int r = 0; r < out_height; ++r) { 
            for (int c = 0; c < out_width; ++c) { 
                // Defn the current slice of the input volume
                size_t vert_start = r * stride;
                size_t vert_end = vert_start + kernel_size;
                size_t horiz_start = c * stride;
                size_t horiz_end = horiz_start + kernel_size;
                
                // Extract the 3D slice from the input
                Tensor3D input_slice = createTensor3D(in_depth, kernel_size, kernel_size);
                 for(int d=0; d < in_depth; ++d) {
                     for(int i = vert_start; i < vert_end; ++i) {
                         for(int j = horiz_start; j < horiz_end; ++j) {
                              input_slice[d][i - vert_start][j - horiz_start] = input[d][i][j];
                         }
                     }
                 }
                // Perform convolution for this step 
                output[f][r][c] = convolveSingleStep(input_slice, filters[f], biases[f]);
            }
        }
    }
    last_output = output; 
    return output;
}

// Convolution Backpropagation

// 1. dL/dInput: Gradient flowing back to the previous layer.
// 2. dL/dFilter: Gradient for updating the filter weights. 
// 3. dL/dBias: Gradient for updating biases. Simpler, just sum gradients for that filter.
Tensor3D ConvLayer::backward(const Tensor3D& output_gradient, float learning_rate) {
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D dL_dInput = createTensor3D(in_depth, in_height, in_width, 0.0f); 
    vector<Tensor3D> dL_dFilters = vector<Tensor3D>(num_filters, createTensor3D(in_depth, kernel_size, kernel_size, 0.0f));
    Tensor1D dL_dBiases = Tensor1D(num_filters, 0.0f);

    // Loop through the output volume dimensions (where we have gradients)
    for (int f = 0; f < num_filters; ++f) {
        for (int r = 0; r < out_height; ++r) {
            for (int c = 0; c < out_width; ++c) {
                float grad_out = output_gradient[f][r][c];
                dL_dBiases[f] += grad_out;

                size_t vert_start = r * stride; // Stride assumed 1 here
                size_t horiz_start = c * stride; // Stride assumed 1 here
                
                // Calculate dL/dFilter and dL/dInput contribution for this gradient element
                for (int d = 0; d < in_depth; ++d) {
                    for (int kr = 0; kr < kernel_size; ++kr) {
                        for (int kc = 0; kc < kernel_size; ++kc) {
                            size_t input_row = vert_start + kr;
                            size_t input_col = horiz_start + kc;
                            if (input_row < in_height && input_col < in_width) { 
                                dL_dFilters[f][d][kr][kc] += last_input[d][input_row][input_col] * grad_out;
                                dL_dInput[d][input_row][input_col] += filters[f][d][kr][kc] * grad_out;
                            }
                        }
                    }
                }
            }
        }
    }
    // Update filters and biases using SGD (Simple Stochastic Gradient Descent)
    for (int f = 0; f < num_filters; ++f) {
        biases[f] -= learning_rate * dL_dBiases[f];
        for (int d = 0; d < in_depth; ++d) {
            for (int kr = 0; kr < kernel_size; ++kr) {
                for (int kc = 0; kc < kernel_size; ++kc) {
                    filters[f][d][kr][kc] -= learning_rate * dL_dFilters[f][d][kr][kc];
                }
            }
        }
    }

    return dL_dInput; 
}

// ReLULayer
ReLULayer::ReLULayer(size_t input_depth, size_t input_height, size_t input_width) {
    input_shape = {input_depth, input_height, input_width};
    output_shape = input_shape;
}

// ReLU activation fn: f(x) = max(0, x)
Tensor3D ReLULayer::forward(const Tensor3D& input) {
    last_input = input; 
    auto [depth, height, width] = input_shape;

    Tensor3D output = createTensor3D(depth, height, width);
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                output[d][r][c] = max(0.0f, input[d][r][c]); 
            }
        }
    }
     last_output = output; 
    return output;
}

Tensor3D ReLULayer::backward(const Tensor3D& output_gradient, float learning_rate) {
    auto [depth, height, width] = input_shape;

    Tensor3D dL_dInput = createTensor3D(depth, height, width);
    // cout<<"Hello 3"<<endl;
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                // cout<<(last_input[d][r][c] > 0.0f)<<endl;
                dL_dInput[d][r][c] = (last_input[d][r][c] > 0.0f) ? output_gradient[d][r][c] : 0.0f;
            }
        }
    }
    // cout<<"chrck"<<endl;
    return dL_dInput;
}

// MaxPoolingLayer
MaxPoolingLayer::MaxPoolingLayer(size_t input_depth, size_t input_height, size_t input_width,
                                     size_t pool_size, size_t stride)
    : pool_size(pool_size), stride(stride)
{
     input_shape = {input_depth, input_height, input_width};

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;
    output_shape = {input_depth, output_height, output_width}; 

    // Initialize storage for max indices
    max_indices.resize(input_depth);
    for(int d=0; d< input_depth; ++d){
         max_indices[d].resize(output_height, vector<pair<size_t, size_t>>(output_width)); 
    }
}


Tensor3D MaxPoolingLayer::forward(const Tensor3D& input) {
    last_input = input; 
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D output = createTensor3D(out_depth, out_height, out_width);

    // cout<<"hello 4"<<endl;
    for (int d = 0; d < in_depth; ++d) {
        for (int r = 0; r < out_height; ++r) {
            for (int c = 0; c < out_width; ++c) {
                // dfining the current pooling window in the ip
                size_t vert_start = r * stride;
                size_t vert_end = vert_start + pool_size;
                size_t horiz_start = c * stride;
                size_t horiz_end = horiz_start + pool_size;

                float max_val = -numeric_limits<float>::infinity(); 
                size_t max_r = 0, max_c = 0; 

                for (int i = vert_start; i < vert_end; ++i) {
                    for (int j = horiz_start; j < horiz_end; ++j) {
                           if (i < in_height && j < in_width) { 
                                if (input[d][i][j] > max_val) {
                                     max_val = input[d][i][j];
                                     max_r = i;
                                     max_c = j;
                                }
                           } 
                    }
                }
                output[d][r][c] = max_val;
                max_indices[d][r][c] = {max_r, max_c}; 
            }
        }
    }
    // cout<<"check"<<endl;
    last_output = output;
    return output;
}


Tensor3D MaxPoolingLayer::backward(const Tensor3D& output_gradient, float learning_rate) {
    // didnt use learning_rate as pooling has no learnable param
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D dL_dInput = createTensor3D(in_depth, in_height, in_width, 0.0f); 

    for (int d = 0; d < out_depth; ++d) {
        for (int r = 0; r < out_height; ++r) {
            for (int c = 0; c < out_width; ++c) {
                size_t max_r = max_indices[d][r][c].first;
                size_t max_c = max_indices[d][r][c].second;
                 if (max_r < in_height && max_c < in_width) { 
                     dL_dInput[d][max_r][max_c] += output_gradient[d][r][c]; 
                 } 
            }
        }
    }
    // cout<<"hello 6"<<endl;
    return dL_dInput;
}

//FlattenLayer implementation
FlattenLayer::FlattenLayer(size_t input_depth, size_t input_height, size_t input_width) {
     input_shape = {input_depth, input_height, input_width};
     size_t flattened_size = input_depth * input_height * input_width;
     // we show flattened output as [flattened_size][1][1]
     output_shape = {flattened_size, 1, 1};
}

Tensor3D FlattenLayer::forward(const Tensor3D& input) {
    last_input = input; 
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D output = createTensor3D(out_depth, out_height, out_width);
    int k = 0; 
    for (int d = 0; d < in_depth; ++d) {
        for (int r = 0; r < in_height; ++r) {
            for (int c = 0; c < in_width; ++c) {
                 output[k++][0][0] = input[d][r][c];
            }
        }
    }
     last_output = output; 
    return output;
}

Tensor3D FlattenLayer::backward(const Tensor3D& output_gradient, float learning_rate) {
    auto [in_depth, in_height, in_width] = input_shape;
    auto [out_depth, out_height, out_width] = output_shape; 

    Tensor3D dL_dInput = createTensor3D(in_depth, in_height, in_width); 
    int k = 0; 
    for (int d = 0; d < in_depth; ++d) {
        for (int r = 0; r < in_height; ++r) {
            for (int c = 0; c < in_width; ++c) {
                  dL_dInput[d][r][c] = output_gradient[k++][0][0];
            }
        }
    }
    return dL_dInput;
}

// DenseLayer (aka Fully Connected Layer) implementatn
DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size)
{
    input_shape = {input_size, 1, 1};
    output_shape = {output_size, 1, 1};

    weights = createTensor2D(output_size, input_size);
    initializeTensor2D(weights, -0.1f, 0.1f); 

    biases = Tensor1D(output_size, 0.0f);
}


Tensor3D DenseLayer::forward(const Tensor3D& input) {
    last_input = input; 

    Tensor3D output = createTensor3D(output_size, 1, 1, 0.0f);

    for (int j = 0; j < output_size; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += weights[j][i] * input[i][0][0];
        }
        output[j][0][0] = sum + biases[j];
    }

    last_output = output; 
    return output;
}

Tensor3D DenseLayer::backward(const Tensor3D& output_gradient, float learning_rate) {
    Tensor3D dL_dInput = createTensor3D(input_size, 1, 1, 0.0f); 
    Tensor2D dL_dWeights = createTensor2D(output_size, input_size, 0.0f); 
    Tensor1D dL_dBiases = Tensor1D(output_size, 0.0f); 

    // Calculating gradnt
    for (int j = 0; j < output_size; ++j) {
        float grad_out_j = output_gradient[j][0][0];
        dL_dBiases[j] = grad_out_j;

        for (int i = 0; i < input_size; ++i) {
            dL_dWeights[j][i] = last_input[i][0][0] * grad_out_j;
            dL_dInput[i][0][0] += weights[j][i] * grad_out_j;
        }
    }

    // Update weights and biases using SGD (Simple Stochastic Gradient Descent)
    // cout<<"hello 8"<<endl;
    for (int j = 0; j < output_size; ++j) {
        biases[j] -= learning_rate * dL_dBiases[j];
        for (int i = 0; i < input_size; ++i) {
            weights[j][i] -= learning_rate * dL_dWeights[j][i];
        }
    }
    // cout<<"chck"<<endl;
    return dL_dInput; 
}

SoftmaxLayer::SoftmaxLayer(size_t input_size) : input_size(input_size) {
    // Input is num of classes 
    input_shape = {input_size, 1, 1};
    // Output is probabilities
    output_shape = {input_size, 1, 1};
}

Tensor3D SoftmaxLayer::forward(const Tensor3D& input) {
     last_input = input; 

     Tensor3D output = createTensor3D(input_size, 1, 1);
     Tensor1D exp_values(input_size);
     float max_logit = -numeric_limits<float>::infinity(); 

     for(int i=0; i < input_size; ++i) {
          if (input[i][0][0] > max_logit) {
               max_logit = input[i][0][0];
          }
     }

     float sum_exp = 0.0f;
     for (int i = 0; i < input_size; ++i) {
          exp_values[i] = exp(input[i][0][0] - max_logit); 
          sum_exp += exp_values[i];
     }

      if (sum_exp == 0.0f) sum_exp = 1e-9; 

     for (int i = 0; i < input_size; ++i) {
          output[i][0][0] = exp_values[i] / sum_exp;
     }

     last_output = output; 
     return output;
}

Tensor3D SoftmaxLayer::backward(const Tensor3D& output_gradient, float learning_rate) {
     return output_gradient; 
}

Tensor3D SoftmaxLayer::calculateInitialGradient(const Tensor1D& true_label_one_hot) {
    Tensor3D gradient = createTensor3D(input_size, 1, 1);
    for (int i = 0; i < input_size; ++i) {
         gradient[i][0][0] = last_output[i][0][0] - true_label_one_hot[i];
    }
    return gradient;
}