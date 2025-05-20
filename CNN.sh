#!/bin/bash

# Compile the C++ source files with OpenCV support, suppressing all warnings
g++ main.cpp cnn.cpp cnn_layers.cpp image_loader.cpp \
    -o cnn_classifier \
    -std=c++17 \
    -w \
    -O2 \
    `pkg-config --cflags --libs opencv4`

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running cnn_classifier..."
    ./cnn_classifier
else
    echo "Compilation failed."
fi