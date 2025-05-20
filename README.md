# CNN implementation in C++ and comparison with non-CNN implementations

[Brief Project Description - e.g., This project explores image processing/classification using a custom CNN implemented in C++ and compares its performance against traditional non-CNN computer vision techniques, also implemented in C++. Compilation and execution are handled directly via shell scripts.]

## Prerequisites

Before you begin, ensure you have the following installed and accessible from your command line environment:

1.  **A C++ Compiler:**
    * **Linux:** `g++` (from `build-essential` package) or `clang++`.
    * **macOS:** `clang++` (from Xcode Command Line Tools).
    * **Windows:** `g++` (via MinGW/MSYS2) or `cl.exe` (from Visual Studio Build Tools configured for command-line use).
    * The compiler command must be available in your system's PATH.

2.  **OpenCV Development Libraries (Headers and Libraries):**
    * You need the C++ header files (`.hpp`, `.h`) and compiled library files (`.so`, `.a`, `.dylib`, `.lib`) for OpenCV (version 4.x recommended).
    * See the installation section below. The compiler invoked by the scripts **must** be able to locate these.

3.  **A Bash-compatible Shell:**
    * Required to execute the `.sh` scripts.
    * Standard on Linux and macOS.
    * Available on Windows through environments like Git Bash (recommended), Windows Subsystem for Linux (WSL), or Cygwin.

4.  **Git:** (Optional, for cloning the repository).

## Installing OpenCV for C++ Development

You need to install the OpenCV library so that the C++ compiler can find its header files for compilation and its library files for linking. The method depends on your OS:

* **Debian/Ubuntu Linux:**
    ```bash
    sudo apt-get update
    sudo apt-get install libopencv-dev
    ```
    *(Installs headers/libs into standard system paths like `/usr/include/opencv4` and `/usr/lib/`)*

* **Fedora Linux:**
    ```bash
    sudo dnf install opencv-devel
    ```
    *(Installs into standard system paths)*

* **macOS (using Homebrew):**
    ```bash
    brew install opencv
    ```
    *(Homebrew typically configures paths correctly for compilers invoked from the terminal)*

* **Windows:**
    * **Option 1 (Easier Path Setup): vcpkg**
        * Install vcpkg: [https://github.com/microsoft/vcpkg](https://github.com/microsoft/vcpkg)
        * Install OpenCV: `vcpkg install opencv4[core,imgproc,highgui]` (add other modules as needed, e.g., `dnn` if used).
        * You might need to use `vcpkg integrate install` or manually adjust script compile/link flags to point to the vcpkg installation directories.
    * **Option 2 (Manual Setup): Pre-built Libraries**
        * Download Windows pack from OpenCV Releases: [https://opencv.org/releases/](https://opencv.org/releases/)
        * Extract (e.g., to `C:\opencv`).
        * **Crucially:** You will likely **need to modify the compile/link commands inside `CNN.sh` and `nonCNN.sh`** to explicitly add:
            * Include paths (e.g., `-IC:/opencv/build/include` for g++, `/I"C:\opencv\build\include"` for MSVC cl.exe).
            * Library paths (e.g., `-LC:/opencv/build/x64/vc16/lib` for g++, `/LIBPATH:"C:\opencv\build\x64\vc16\lib"` for MSVC cl.exe). Adjust `vc16` based on your VS version.
        * Add the OpenCV `bin` directory (e.g., `C:\opencv\build\x64\vc16\bin`) to your system's PATH environment variable for finding DLLs at runtime.

**Important Note on Paths:** If OpenCV is installed in a location where the compiler doesn't automatically look, the compile commands inside `CNN.sh` and `nonCNN.sh` might fail (e.g., "header file not found", "linker error"). You **may need to manually edit these scripts** to add the correct flags (`-I<path_to_include>`, `-L<path_to_lib>`) pointing to your specific OpenCV installation directories.

## Compiling and Running the Implementations

This project is compiled and executed directly using the provided shell scripts. There is no separate configuration or build step (like `cmake` or `make`). Each script contains the necessary compiler commands.

**Important:** Run these scripts from the project's **root directory** (the directory containing the `.sh` files and your C++ source code).

1.  **Compile and Run the CNN Implementation:**
    This script will invoke the C++ compiler (e.g., `g++` or `clang++` as specified *inside* the script) to build the CNN-related source files, link them against the required OpenCV libraries, and then, if compilation is successful, run the resulting executable.

    ```bash
    bash CNN.sh
    ```
    *(On Linux/macOS, you might need to grant execute permission first using `chmod +x CNN.sh`, then you can run it directly with `./CNN.sh`)*

2.  **Compile and Run the Non-CNN Implementation(s):**
    This script performs the same compile-and-run process but for the non-CNN C++ source files used for comparison.

    ```bash
    bash nonCNN.sh
    ```
    *(On Linux/macOS, you might need execute permission: `chmod +x nonCNN.sh`, then run with `./nonCNN.sh`)*

**Troubleshooting:**

* **"Command not found" (g++/clang++/cl.exe):** Ensure your C++ compiler is installed and its directory is included in your system's PATH environment variable.
* **"Permission Denied" (running ./script.sh):** Use `chmod +x script.name.sh` to make the script executable.
* **Compilation Errors (e.g., "opencv2/.../core.hpp: No such file or directory"):**
    * Verify that the OpenCV Development Libraries (including headers) are correctly installed.
    * Check if your installation location is standard. If not, you likely need to edit the `CNN.sh` and `nonCNN.sh` scripts to add the appropriate `-I<path_to_opencv_include>` flag to the compiler command within the script.
* **Linker Errors (e.g., "undefined reference to `cv::imread(..."`):**
    * Verify that the OpenCV library files are installed.
    * Check if your installation location is standard. If not, you likely need to edit the scripts to add the `-L<path_to_opencv_lib>` flag to the compiler command.
    * Ensure the script correctly links the necessary OpenCV modules using `-l` flags (e.g., `-lopencv_core -lopencv_imgproc -lopencv_highgui`). The required modules depend on the OpenCV functions your code uses.
* **Runtime Errors (Windows: "DLL not found"):** Make sure the directory containing the OpenCV `.dll` files (e.g., `C:\opencv\build\x64\vc16\bin`) is in your system's PATH environment variable.

---

*(Optional: Add more details about the specific non-CNN methods used, dataset information, how to interpret results, etc.)*