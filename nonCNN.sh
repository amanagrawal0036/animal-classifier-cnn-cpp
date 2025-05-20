#!/bin/bash

# --- Script to Compile and Run the C++ Scratch Image Classifier ---
OPENCV_INCLUDE_DIR="/usr/include/opencv4"
OPENCV_LIB_DIR="/usr/lib/x86_64-linux-gnu"


# --- Script Variables ---
CXX="g++"                      # C++ Compiler (can be clang++)
SOURCE_FILE="nonCNN.cpp"       # Your C++ source file name
EXECUTABLE_NAME="image_classifier_scratch" # Output program name
CPP_STANDARD="-std=c++17"      # Use C++17 for <filesystem>
OPENCV_LIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" # OpenCV libraries needed
COMPILE_FLAGS="-O2"            # Optimization flag (e.g., -O2, -O3, or -g for debug)
# Optional: For Linux, uncomment and set if needed to help find .so files at runtime
# LINKER_FLAGS="-Wl,-rpath,${OPENCV_LIB_DIR}"
LINKER_FLAGS=""                # No extra linker flags by default

# --- Script Logic ---

# Check if source file exists
if [ ! -f "${SOURCE_FILE}" ]; then
    echo "Error: Source file '${SOURCE_FILE}' not found in the current directory."
    exit 1
fi

# Check if OpenCV paths were set (basic check - they might still be wrong)
# Allow default paths used by package managers
# if [[ "${OPENCV_INCLUDE_DIR}" == "/path/to/your/opencv/include" || "${OPENCV_LIB_DIR}" == "/path/to/your/opencv/lib" ]]; then
#     echo "Warning: OpenCV paths might not be set correctly. Using default system paths."
#     # You might remove the exit here if system paths are likely correct
#     # exit 1
# fi

echo "--- Compiling ${SOURCE_FILE} ---"

# --- Execute the compilation command directly ---
# Removed the COMPILE_COMMAND variable and eval
# Quote arguments that might contain spaces (paths, filenames)
# Use standard spaces for indentation
"${CXX}" "${SOURCE_FILE}" \
  ${CPP_STANDARD} \
  -I"${OPENCV_INCLUDE_DIR}" \
  -L"${OPENCV_LIB_DIR}" \
  -o "${EXECUTABLE_NAME}" \
  ${OPENCV_LIBS} \
  ${COMPILE_FLAGS} \
  ${LINKER_FLAGS}

# Check if compilation was successful (immediately after the command)
# $? holds the exit status of the last command
if [ $? -ne 0 ]; then
    echo "--- Compilation failed. ---"
    exit 1
fi
# --- End of modified compilation block ---

echo "--- Compilation successful: ${EXECUTABLE_NAME} ---"
echo ""

# Check if the executable was created
if [ ! -x "${EXECUTABLE_NAME}" ]; then # Check for executable permission as well
    echo "Error: Executable '${EXECUTABLE_NAME}' was not created or is not executable."
    exit 1
fi

echo "--- Running ${EXECUTABLE_NAME} ---"
echo "Ensure image_data.csv and image files are in this directory or accessible."
echo ""

# Run the executable
./"${EXECUTABLE_NAME}"

# Check the exit status of the program
EXIT_STATUS=$?
if [ ${EXIT_STATUS} -ne 0 ]; then
    echo ""
    echo "--- Program exited with error status ${EXIT_STATUS}. ---"
    exit 1
fi

echo ""
echo "--- Program finished successfully. ---"
exit 0