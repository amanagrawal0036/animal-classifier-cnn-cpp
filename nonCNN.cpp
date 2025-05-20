#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm> 
#include <map>
#include <iomanip> 
#include <chrono>   
#include <filesystem> 
#include <cmath>      
#include <limits>    
// OpenCV Headers (ONLY for Image I/O and Resize)
#include <opencv2/core.hpp>     // For cv::Mat, cv::Size
#include <opencv2/imgproc.hpp>  // For cv::resize, cv::INTER_AREA
#include <opencv2/imgcodecs.hpp>// For cv::imread, cv::IMREAD_GRAYSCALE

using namespace std;

// --- Type Definitions ---
using Matrix = vector<vector<double>>;
using Vector = vector<double>;
using Labels = vector<int>;

// --- Configuration ---
const string CSV_FILE = "image_data.csv";
const int IMG_WIDTH = 64;
const int IMG_HEIGHT = 64;
const double TEST_SIZE = 0.20;
const int RANDOM_STATE = 42; // Seed for shuffling
const int N_FEATURES = IMG_WIDTH * IMG_HEIGHT; // Derived feature count
const double LEARNING_RATE = 0.01; // Learning rate for Logistic Regression
const int MAX_ITERATIONS = 1000;   // Iterations for Logistic Regression Gradient Descent
const double EPSILON = 1e-8;       // Small value to prevent division by zero / log(0)
// --- End Configuration ---

// --- Basic Linear Algebra Helpers (Scratch Implementation) ---

// Dot product of two vectors
double dot_product(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        cerr << "Error: Vectors must have the same size for dot product." << endl;
        return 0.0; // Return 0 on error
    }
    double result = 0.0;

    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Vector subtraction: a - b
Vector vector_subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        cerr << "Error: Vectors must have the same size for subtraction." << endl;
        return {}; // Return empty vector on error
    }
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Vector addition: a + b
Vector vector_add(const Vector& a, const Vector& b) {
     if (a.size() != b.size()) {
         cerr << "Error: Vectors must have the same size for addition." << endl;
         return {}; // Return empty vector on error
    }
    Vector result(a.size());
      
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}


// Scalar multiplication: scalar * vec
Vector vector_scalar_multiply(double scalar, const Vector& vec) {
    Vector result(vec.size());
      
    for (int i = 0; i < vec.size(); ++i) {
        result[i] = scalar * vec[i];
    }
    return result;
}

// Sum of vector elements
double sum_vector(const Vector& vec) {
      double sum = 0.0;
      for(double val : vec) {
          sum += val;
      }
      return sum;
}

// Matrix-vector multiplication: M * v
Vector matrix_vector_multiply(const Matrix& M, const Vector& v) {
    if (M.empty() || M[0].empty() || M[0].size() != v.size()) {
        cerr << "Error: Matrix and vector dimensions mismatch for multiplication." << endl;
        return {}; // Return empty vector on error
    }
      
    int num_rows = M.size();
    Vector result(num_rows);
    for (int i = 0; i < num_rows; ++i) {
        result[i] = dot_product(M[i], v); // dot_product handles its own check now
    }
    return result;
}

// Transposed Matrix-vector multiplication: M^T * v
// Computes the dot product of each *column* of M with v
Vector matrix_transpose_vector_multiply(const Matrix& M, const Vector& v) {
    if (M.empty() || M.size() != v.size()) {
          cerr << "Error: Matrix and vector dimensions mismatch for transposed multiplication." << endl;
          return {}; // Return empty vector on error
    }
      
    int num_rows = M.size();
    int num_cols = M[0].size(); // Number of features, length of output vector
    Vector result(num_cols, 0.0);
    for(int j = 0; j < num_cols; ++j) { // Iterate through columns of M (features)
        for(int i = 0; i < num_rows; ++i) { // Iterate through rows of M (samples)
              result[j] += M[i][j] * v[i];
        }
    }
    return result;
}

// --- Data Structures ---
struct Dataset {
    Matrix features; // Using vector<vector<double>>
    Labels labels;
    vector<string> image_paths; // Original paths for reference
    vector<int> unique_classes;      // Store unique class labels found
};

struct SplitDataset {
    Matrix X_train, X_test;
    Labels y_train, y_test;
};

// --- StandardScaler (Scratch Implementation) ---
class StandardScaler {
private:
    Vector feature_means;
    Vector feature_stddevs;
    bool fitted = false;

public:
    StandardScaler() = default;

    // Calculate mean and stddev from training data
    void fit(const Matrix& X_train) {
        if (X_train.empty() || X_train[0].empty()) {
            cerr << "Error: Cannot fit StandardScaler on empty data." << endl;
            fitted = false; // Ensure fitted is false
            return;
        }
          
        int n_samples = X_train.size();
        int n_features = X_train[0].size();

        feature_means.assign(n_features, 0.0);
        feature_stddevs.assign(n_features, 0.0);
        Vector feature_sq_means(n_features, 0.0); // For calculating variance

        // Calculate sum and sum of squares for each feature
        for (const auto& sample : X_train) {
            if (sample.size() != n_features) {
                cerr << "Error: Inconsistent feature size in training data during fit." << endl;
                fitted = false;
                return;  
            }
              
            for (int j = 0; j < n_features; ++j) {
                feature_means[j] += sample[j];
                feature_sq_means[j] += sample[j] * sample[j];
            }
        }

        // Finalize mean and calculate standard deviation
          
        for (int j = 0; j < n_features; ++j) {
            double mean = feature_means[j] / n_samples;
            double sq_mean = feature_sq_means[j] / n_samples;
            double variance = sq_mean - (mean * mean);
            // Ensure variance is non-negative (can happen due to floating point errors)
             variance = max(0.0, variance); // Use max from <algorithm> via namespace std
            feature_means[j] = mean;
            feature_stddevs[j] = sqrt(variance); // Use sqrt from <cmath> via namespace std
            // Add epsilon to stddev to prevent division by zero during transform
            if (feature_stddevs[j] < EPSILON) {
                feature_stddevs[j] = EPSILON;
            }
        }
        fitted = true;
    }

    // Apply scaling transform (X - mean) / stddev
    Matrix transform(const Matrix& X) {
        if (!fitted) {
            cerr << "Error: StandardScaler must be fitted before transforming data." << endl;
            return {}; // Return empty matrix on error
        }
        if (X.empty()) return {}; // Return empty if input is empty

          
        int n_samples = X.size();
        int n_features = X[0].size();
        if (n_features != feature_means.size()) {
            cerr << "Error: Data to transform has different number of features than fitted data." << endl;
            return {}; // Return empty matrix on error
        }

        Matrix X_scaled(n_samples, Vector(n_features));
          
        for (int i = 0; i < n_samples; ++i) {
             if (X[i].size() != n_features) {
                 cerr << "Error: Inconsistent feature size in data to transform." << endl;
                 return {}; // Return empty matrix on error
             }
               
            for (int j = 0; j < n_features; ++j) {
                X_scaled[i][j] = (X[i][j] - feature_means[j]) / feature_stddevs[j];
            }
        }
        return X_scaled;
    }

     // Convenience function to fit and then transform
    Matrix fit_transform(const Matrix& X_train) {
        fit(X_train);
        // Check if fit was successful before transforming
        if (!fitted) {
            return {}; // Return empty if fit failed
        }
        return transform(X_train);
    }

    // Getters for diagnostics if needed
    Vector get_means() const { return feature_means; }
    Vector get_stddevs() const { return feature_stddevs; }
};


// --- Logistic Regression (Scratch Implementation - One-vs-Rest) ---
class LogisticRegressionOvR {
private:
    vector<int> classes_;           // List of unique class labels
    Matrix weights_;                // One weight vector per class [n_classes x n_features]
    Vector biases_;                 // One bias per class [n_classes]
    double learning_rate_;
    int iterations_;
    bool fitted = false;

    // Sigmoid function
    double sigmoid(double z) const {
        // Add clipping to prevent exp overflow/underflow
        if (z > 35.0) return 1.0; // exp(35) is large enough
        if (z < -35.0) return 0.0; // exp(-35) is very small
        return 1.0 / (1.0 + exp(-z));
    }

     // Helper to compute z = w*x + b
    double compute_z(const Vector& x, const Vector& w, double b) const {
         return dot_product(x, w) + b;
    }

public:
    LogisticRegressionOvR(double lr = 0.01, int iterations = 1000)
        : learning_rate_(lr), iterations_(iterations) {}

    void fit(const Matrix& X_train, const Labels& y_train, const vector<int>& unique_classes) {
        if (X_train.empty() || X_train.size() != y_train.size()) {
             cerr << "Error: Invalid training data for Logistic Regression fit." << endl;
             fitted = false; // Mark as not fitted
             return;  
        }
        classes_ = unique_classes;
          
        int n_samples = X_train.size();
        int n_features = X_train[0].size();
        int n_classes = classes_.size();

        // Initialize weights and biases (one set per class for OvR)
        weights_.assign(n_classes, Vector(n_features, 0.0));
        biases_.assign(n_classes, 0.0);

        // Train one binary classifier for each class
          
        for (int c_idx = 0; c_idx < n_classes; ++c_idx) {
            int current_class = classes_[c_idx];
            cout << "  Training classifier for class " << current_class << "..." << endl;

            // Create binary labels for this classifier (1 for current_class, 0 otherwise)
            Vector y_binary(n_samples);
              
            for(int i = 0; i < n_samples; ++i) {
                y_binary[i] = (y_train[i] == current_class) ? 1.0 : 0.0;
            }

            // Get references to the weights and bias for this specific class
            Vector& w = weights_[c_idx];
            double& b = biases_[c_idx];

            // Gradient Descent for this binary classifier
            for (int iter = 0; iter < iterations_; ++iter) {
                // Calculate predictions (hypothesis) h = sigmoid(X*w + b)
                Vector h(n_samples);
                  
                for(int i = 0; i < n_samples; ++i) {
                    h[i] = sigmoid(compute_z(X_train[i], w, b));
                }

                // Calculate error = h - y_binary
                Vector error = vector_subtract(h, y_binary);
                // Check if subtraction failed
                if (error.empty()) {
                    cerr << "Error during error calculation in fit." << endl;
                    fitted = false;
                    return;
                }

                // Calculate gradients
                // dw = (1/m) * X^T * error
                Vector dw = matrix_transpose_vector_multiply(X_train, error);
                 // Check if multiplication failed
                if (dw.empty()) {
                     cerr << "Error calculating dw gradient in fit." << endl;
                     fitted = false;
                     return;
                 }
                dw = vector_scalar_multiply(1.0 / n_samples, dw);

                // db = (1/m) * sum(error)
                double db = sum_vector(error) / n_samples;

                // Update weights and bias
                Vector scaled_dw = vector_scalar_multiply(learning_rate_, dw);
                w = vector_subtract(w, scaled_dw);
                b = b - learning_rate_ * db;
                 // Check if subtraction failed
                if (w.empty()) {
                    cerr << "Error updating weights in fit." << endl;
                    fitted = false;
                    return;
                }


                // Optional: Print cost function (Log Loss) periodically
                 if ((iter + 1) % 100 == 0) {
                       double cost = 0.0;
                         
                       for(int i=0; i<n_samples; ++i) {
                           // Add epsilon to prevent log(0)
                           double h_i = max(EPSILON, min(1.0 - EPSILON, h[i])); // Use max/min from <algorithm>
                           if (y_binary[i] == 1.0) {
                               cost -= log(h_i); // Use log from <cmath>
                           } else {
                               cost -= log(1.0 - h_i); // Use log from <cmath>
                           }
                       }
                       cost /= n_samples;
                       //cout << "     Iter: " << iter + 1 << ", Cost: " << cost << endl;
                 }
            } // End gradient descent iterations
        } // End loop over classes
        fitted = true;
         cout << "  OvR Training Complete." << endl;
    }

    // Predict class probabilities for each class (OvR)
    Matrix predict_proba(const Matrix& X) const {
        if (!fitted) {
            cerr << "Error: Model not fitted yet. Cannot predict probabilities." << endl;
            return {}; // Return empty matrix
        }
        if (X.empty()) return {};

          
        int n_samples = X.size();
        int n_classes = classes_.size();
        Matrix probabilities(n_samples, Vector(n_classes));

          
        for (int i = 0; i < n_samples; ++i) {
              
            for (int c_idx = 0; c_idx < n_classes; ++c_idx) {
                probabilities[i][c_idx] = sigmoid(compute_z(X[i], weights_[c_idx], biases_[c_idx]));
            }
        }
        return probabilities;
    }

    // Predict the class label (the one with the highest probability)
    Labels predict(const Matrix& X) const {
        Matrix probabilities = predict_proba(X);
        // Check if predict_proba failed
        if (probabilities.empty() && !X.empty()) {
            cerr << "Error: Probability prediction failed. Cannot predict labels." << endl;
            return {}; // Return empty labels
        }
        if (X.empty()) return {}; // Handle empty input case

        Labels predictions(X.size());

          
        for (int i = 0; i < X.size(); ++i) {
            // Find the index (and thus class) with the maximum probability
            auto max_it = max_element(probabilities[i].begin(), probabilities[i].end()); // Use max_element from <algorithm>
              
            int predicted_idx = distance(probabilities[i].begin(), max_it); // Use distance from <iterator> via namespace std
            // Check if predicted index is valid
            if (predicted_idx >= 0 && predicted_idx < classes_.size()) {
                 predictions[i] = classes_[predicted_idx];
            } else {
                 cerr << "Error: Invalid predicted index " << predicted_idx << " during prediction." << endl;
                 // Assign a default value or handle error - let's assign first class as fallback
                 if (!classes_.empty()) {
                     predictions[i] = classes_[0];
                 } else {
                     predictions[i] = -1; // Indicate error if no classes exist
                 }
            }
        }
        return predictions;
    }
};


bool loadAndPreprocessDataScratch(const string& csv_path, int img_height, int img_width, Dataset& data) {
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Error: CSV file not found at " << csv_path << endl;
        return false;
    }

    string line;
    string image_path_col_name, label_col_name;
    bool header_read = false;

    vector<Vector> feature_vectors; // Store as vector of doubles
    vector<int> label_vector;
    map<int, int> class_counts;

    filesystem::path csv_fs_path(csv_path);
    // Get the absolute path to the directory containing the CSV file
    filesystem::path base_path = filesystem::absolute(csv_fs_path).parent_path();
    cout << "Base path for images directory assumed: " << base_path << endl;

    // Define the subdirectory name where images are located
    const string image_subdir = "images"; // <-- Define the subdirectory name here

    int processed_count = 0;
    int error_count = 0;
    int row_index = 0; // Track row number for better error messages

    while (getline(file, line)) {
        row_index++;
        // Skip empty lines or lines with only whitespace
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == string::npos) continue;

        stringstream ss(line);
        string cell;
        vector<string> row_values;
        while (getline(ss, cell, ',')) {
             // Trim leading/trailing whitespace from cell
             cell.erase(0, cell.find_first_not_of(" \t\n\r"));
             cell.erase(cell.find_last_not_of(" \t\n\r") + 1);
            row_values.push_back(cell);
        }

        // Basic check for sufficient columns
        if (row_values.size() < 2) {
             cerr << "Warning: Row " << row_index << " has less than 2 columns. Skipping." << endl; error_count++; continue;
        }

        // Read header row (assuming first row is header)
        if (!header_read) {
            image_path_col_name = row_values[0]; // e.g., "Image Name"
            label_col_name = row_values[1];      // e.g., "Label"
            header_read = true;
            cout << "Reading CSV with assumed columns: '" << image_path_col_name << "', '" << label_col_name << "'" << endl;
            continue; // Skip processing the header row
        }

        string image_filename = row_values[0]; // e.g., "7319.jpg"
        string label_str = row_values[1];      // e.g., "2"
        int label;

        // Validate label format before conversion
        if (label_str.empty() || label_str.find_first_not_of("0123456789-+") != string::npos) {
            cerr << "Warning: Invalid label format '" << label_str << "' in row " << row_index << ". Skipping." << endl; error_count++; continue;
        }

        try {
             label = stoi(label_str); // Convert label string to integer
        } catch (const std::invalid_argument& ia) {
             cerr << "Warning: Invalid label value '" << label_str << "' (not an integer) in row " << row_index << ". Skipping." << ia.what() << endl; error_count++; continue;
        } catch (const std::out_of_range& oor) {
             cerr << "Warning: Label value '" << label_str << "' out of range for int in row " << row_index << ". Skipping." << oor.what() << endl; error_count++; continue;
        }

        filesystem::path img_full_path = base_path / image_subdir / image_filename;

        string img_path_str = img_full_path.string(); // Convert path to string for OpenCV

        // Use OpenCV ONLY to read and resize the image
        cv::Mat img = cv::imread(img_path_str, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Warning: Cannot read or find image at '" << img_path_str << "' (referenced in row " << row_index << "). Skipping." << endl; error_count++; continue;
        }

        cv::Mat img_resized;
        // Use INTER_AREA for downsampling, generally better quality
        cv::resize(img, img_resized, cv::Size(img_width, img_height), 0, 0, cv::INTER_AREA);

        // Manually flatten and convert to vector<double>, normalize pixel values to [0, 1]
        Vector current_feature_vector(img_width * img_height);
        int k = 0;
        for (int r = 0; r < img_resized.rows; ++r) {
            for (int c = 0; c < img_resized.cols; ++c) {
                // Access pixel value and normalize
                current_feature_vector[k++] = static_cast<double>(img_resized.at<uchar>(r, c)) / 255.0;
            }
        }

        // Check if the flattened vector size matches the expected feature size
        // Ensure N_FEATURES is correctly defined (e.g., const int N_FEATURES = img_width * img_height;)
        if (k != N_FEATURES) {
            cerr << "Error: Feature size mismatch (" << k << " vs expected " << N_FEATURES << ") for image " << img_path_str << ". Ensure N_FEATURES is set correctly. Skipping." << endl; error_count++; continue;
        }

        // Store the processed data
        feature_vectors.push_back(current_feature_vector);
        label_vector.push_back(label);
        data.image_paths.push_back(img_path_str); // Store the full path used
        class_counts[label]++;                    // Count occurrences of each class label
        processed_count++;

        // Provide periodic updates
        if ((processed_count + error_count) % 100 == 0 && (processed_count + error_count) > 0) {
            cout << "  Processed: " << processed_count << ", Errors: " << error_count << " (Reading row " << row_index << ")" << endl;
        }
    }
    file.close();

    cout << "\nFinished loading data." << endl;
    cout << "  Total rows processed (excluding header): " << processed_count << endl;
    cout << "  Total errors/skipped rows: " << error_count << endl;

    if (feature_vectors.empty()) {
        cerr << "Error: No valid image data was loaded. Check CSV format, image paths, and file permissions." << endl;
        return false;
    }

    // Assign the collected data to the output struct
    data.features = feature_vectors;
    data.labels = label_vector;

    // Populate unique classes found
    data.unique_classes.clear(); // Ensure it's empty before filling
    for(auto const& [key, val] : class_counts) {
        data.unique_classes.push_back(key);
    }
    sort(data.unique_classes.begin(), data.unique_classes.end()); // Ensure unique classes are sorted

    cout << "Data loaded successfully. Found " << data.unique_classes.size() << " unique classes." << endl;
    cout << "Class counts:" << endl;
    for(const auto& pair : class_counts) {
        cout << "  Class " << pair.first << ": " << pair.second << " samples" << endl;
    }


    return true;
}

// --- Stratified Split (Manual - Same logic as before, adapted for Matrix/Labels) ---
void stratifiedSplitScratch(const Matrix& X, const Labels& y, double test_ratio, int random_seed, SplitDataset& split_data) {
      
    int n_samples = X.size();
    if (n_samples != y.size()) {
        cerr << "Error Split: X and y size mismatch (" << X.size() << " vs " << y.size() << ")." << endl;
        return;
    }
    if (n_samples == 0) {
         cerr << "Error Split: Cannot split empty data." << endl;
         return;
    }


    map<int, vector<int>> class_indices; // Map label to list of indices (use int index)
      
    for (int i = 0; i < n_samples; ++i) {
        class_indices[y[i]].push_back(i);
    }

    vector<int> train_idx, test_idx;
    mt19937 rng(random_seed);

    for (auto const& [label, indices] : class_indices) {
        vector<int> shuffled_indices = indices;
        shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);
          
        int n_test_class = static_cast<int>(round(shuffled_indices.size() * test_ratio));
        // Ensure at least one sample per class in train/test if possible
         if (n_test_class == 0 && shuffled_indices.size() > 1) n_test_class = 1;
         if (n_test_class >= shuffled_indices.size() && shuffled_indices.size() > 0) n_test_class = shuffled_indices.size() - 1; // Use >= and check size > 0
         if (n_test_class < 0) n_test_class = 0; // Ensure non-negative

        // Check bounds before inserting
        if (n_test_class >= 0 && n_test_class <= shuffled_indices.size()) {
             test_idx.insert(test_idx.end(), shuffled_indices.begin(), shuffled_indices.begin() + n_test_class);
             train_idx.insert(train_idx.end(), shuffled_indices.begin() + n_test_class, shuffled_indices.end());
        } else {
            cerr << "Warning: Invalid split index calculated for class " << label << ". Skipping class for split." << endl;
             train_idx.insert(train_idx.end(), shuffled_indices.begin(), shuffled_indices.end());
        }
    }
    shuffle(train_idx.begin(), train_idx.end(), rng);
    shuffle(test_idx.begin(), test_idx.end(), rng);

    // Reserve space for efficiency
    split_data.X_train.reserve(train_idx.size());
    split_data.y_train.reserve(train_idx.size());
    split_data.X_test.reserve(test_idx.size());
    split_data.y_test.reserve(test_idx.size());

    for (int idx : train_idx) {
        if (idx >= 0 && idx < X.size()) { // Add boundary check
            split_data.X_train.push_back(X[idx]);
            split_data.y_train.push_back(y[idx]);
        } else {
             cerr << "Warning: Invalid training index " << idx << " during split assignment." << endl;
        }
    }
    for (int idx : test_idx) {
        if (idx >= 0 && idx < X.size()) { // Add boundary check
            split_data.X_test.push_back(X[idx]);
            split_data.y_test.push_back(y[idx]);
        } else {
            cerr << "Warning: Invalid testing index " << idx << " during split assignment." << endl;
        }
    }
}


// --- Evaluation Metrics (Scratch Implementation) ---
float calculateAccuracyScratch(const Labels& y_true, const Labels& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        if (y_true.size() != y_pred.size()) cerr << "Warning: y_true and y_pred size mismatch in accuracy calculation." << endl;
        return 0.0f;
    }
    int correct_count = 0;
      
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            correct_count++;
        }
    }
    return static_cast<float>(correct_count) / y_true.size();
}

void evaluateModelScratch(const Labels& y_true, const Labels& y_pred, const vector<int>& class_labels) {
      
    int num_classes = class_labels.size();
    if (num_classes == 0) {
        cerr << "Warning: No class labels provided for evaluation." << endl;
        return;
    }
    if (y_true.size() != y_pred.size()) {
         cerr << "Error: y_true and y_pred size mismatch in evaluation." << endl;
         return;
    }
     if (y_true.empty()) {
         cout << "Evaluation skipped: No data points." << endl;
         return;
     }


    // Map class label to its index in the confusion matrix
     map<int, int> label_to_index;
       
     for(int i = 0; i < num_classes; ++i) label_to_index[class_labels[i]] = i;


    // --- Confusion Matrix ---
    Matrix confusion_matrix_counts(num_classes, Vector(num_classes, 0.0)); // Use double for calculations
      
    for (int i = 0; i < y_true.size(); ++i) {
        int true_label = y_true[i];
        int pred_label = y_pred[i];

         if (label_to_index.count(true_label) && label_to_index.count(pred_label)) {
              int true_idx = label_to_index[true_label];
              int pred_idx = label_to_index[pred_label];
              confusion_matrix_counts[true_idx][pred_idx]++;
         } else {
              cerr << "Warning: Encountered unexpected label during evaluation. True: " << true_label << ", Pred: " << pred_label << endl;
         }
    }

    cout << "\nTest Set Confusion Matrix:" << endl;
    cout << setw(10) << "True\\Pred |";
    for(int label : class_labels) cout << setw(8) << label << " |";
    cout << endl << setw(10) << string(10, '-');
      
    for(int i=0; i < num_classes; ++i) cout << "+" << setw(8) << string(8, '-');
    cout << "+" << endl;
      
    for (int i = 0; i < num_classes; ++i) {
        cout << setw(9) << class_labels[i] << " |";
          
        for (int j = 0; j < num_classes; ++j) {
            cout << setw(8) << static_cast<int>(confusion_matrix_counts[i][j]) << " |"; // Print as int
        }
        cout << endl;
    }

    // --- Classification Report ---
    cout << "\nTest Set Classification Report:" << endl;
    cout << setw(12) << "Class" << setw(12) << "Precision" << setw(12) << "Recall" << setw(12) << "F1-Score" << setw(12) << "Support" << endl;
    cout << string(60, '-') << endl;

    double total_support = 0;
    double weighted_precision = 0, weighted_recall = 0, weighted_f1 = 0;
    double macro_precision = 0, macro_recall = 0, macro_f1 = 0;

      
    for (int i = 0; i < num_classes; ++i) { // i corresponds to the class index
        double tp = confusion_matrix_counts[i][i];
        double fp = 0; // Sum column i (excluding TP)
        double fn = 0; // Sum row i (excluding TP)
        double support = 0; // Sum row i (total true instances)

          
        for (int k = 0; k < num_classes; ++k) {
            if (k != i) {
                fp += confusion_matrix_counts[k][i];
                fn += confusion_matrix_counts[i][k];
            }
             support += confusion_matrix_counts[i][k];
        }

        double precision = (tp + fp == 0) ? 0.0 : tp / (tp + fp);
        double recall = (tp + fn == 0) ? 0.0 : tp / (tp + fn); // Also sensitivity
        double f1_score = (precision + recall == 0) ? 0.0 : 2.0 * (precision * recall) / (precision + recall);

        cout << setw(12) << class_labels[i]
             << fixed << setprecision(2) << setw(12) << precision
             << fixed << setprecision(2) << setw(12) << recall
             << fixed << setprecision(2) << setw(12) << f1_score
             << fixed << setprecision(0) << setw(12) << support << endl;

         total_support += support;
         weighted_precision += precision * support;
         weighted_recall += recall * support;
         weighted_f1 += f1_score * support;
         macro_precision += precision;
         macro_recall += recall;
         macro_f1 += f1_score;
    }
     cout << string(60, '-') << endl;

     // Calculate overall accuracy (micro average) from CM
     double overall_tp = 0;
       
     for(int i=0; i<num_classes; ++i) overall_tp += confusion_matrix_counts[i][i];
     double accuracy_cm = (total_support == 0) ? 0.0 : overall_tp / total_support;

     cout << setw(12) << "Accuracy"
          << fixed << setprecision(2) << setw(12) << "" << setw(12) << ""
          << fixed << setprecision(2) << setw(12) << accuracy_cm
          << fixed << setprecision(0) << setw(12) << total_support << endl;

    // Calculate macro averages (unweighted)
    macro_precision /= (num_classes > 0 ? num_classes : 1.0); // Avoid division by zero
    macro_recall /= (num_classes > 0 ? num_classes : 1.0);
    macro_f1 /= (num_classes > 0 ? num_classes : 1.0);
     cout << setw(12) << "Macro Avg"
          << fixed << setprecision(2) << setw(12) << macro_precision
          << fixed << setprecision(2) << setw(12) << macro_recall
          << fixed << setprecision(2) << setw(12) << macro_f1
          << fixed << setprecision(0) << setw(12) << total_support << endl;


    // Calculate weighted averages
    double avg_precision_w = (total_support == 0) ? 0.0 : weighted_precision / total_support;
    double avg_recall_w = (total_support == 0) ? 0.0 : weighted_recall / total_support;
    double avg_f1_w = (total_support == 0) ? 0.0 : weighted_f1 / total_support;
    cout << setw(12) << "Weighted Avg"
          << fixed << setprecision(2) << setw(12) << avg_precision_w
          << fixed << setprecision(2) << setw(12) << avg_recall_w
          << fixed << setprecision(2) << setw(12) << avg_f1_w
          << fixed << setprecision(0) << setw(12) << total_support << endl;
}


// --- Main Script ---
int main() {
    cout << "--- Non-CNN Image Classification Baseline (C++ Scratch Implementation) ---" << endl;
     cout << "Using OpenCV ONLY for Image I/O and Resize." << endl;
     cout << "Split: " << (1.0 - TEST_SIZE) * 100 << "% Training, " << TEST_SIZE * 100 << "% Testing." << endl;

    if (!filesystem::exists(CSV_FILE)) {
        cerr << "Error: CSV file '" << CSV_FILE << "' not found." << endl; return 1;
    }

    // 1. Load and preprocess data
    Dataset all_data;
    auto start_load = chrono::high_resolution_clock::now();
    if (!loadAndPreprocessDataScratch(CSV_FILE, IMG_HEIGHT, IMG_WIDTH, all_data)) {
        cerr << "Exiting due to data loading errors." << endl; return 1;
    }
    auto end_load = chrono::high_resolution_clock::now();
    chrono::duration<double> load_duration = end_load - start_load;
     cout << "Data loading took: " << fixed << setprecision(2) << load_duration.count() << " seconds." << endl;

     cout << "\nData shape: X=[" << all_data.features.size() << "x" << (all_data.features.empty() ? 0 : all_data.features[0].size()) << "], y=[" << all_data.labels.size() << "]" << endl;
    if (all_data.features.empty()) { cerr << "Error: No data loaded." << endl; return 1; }

     cout << "Unique classes found: " << all_data.unique_classes.size() << " -> { ";
     for(int label : all_data.unique_classes) cout << label << " ";
     cout << "}" << endl;


    // 2. Split data
    cout << "\nSplitting data..." << endl;
    SplitDataset split_data;
      
    stratifiedSplitScratch(all_data.features, all_data.labels, TEST_SIZE, RANDOM_STATE, split_data);
    // Check if split was successful by looking at sizes (stratifiedSplitScratch prints errors)
    if (split_data.X_train.empty() || split_data.y_train.empty()) {
         cerr << "Error splitting data. Training set is empty." << endl;
         return 1;
    }

     cout << "Training set: X=[" << split_data.X_train.size() << "x" << N_FEATURES << "], y=[" << split_data.y_train.size() << "]" << endl;
     cout << "Testing set:  X=[" << split_data.X_test.size() << "x" << N_FEATURES << "], y=[" << split_data.y_test.size() << "]" << endl;

    // 3. Feature Scaling
    cout << "\nScaling features (StandardScaler from scratch)..." << endl;
    StandardScaler scaler;
    Matrix X_train_scaled, X_test_scaled;
       
     scaler.fit(split_data.X_train); // Fit ONLY on training data
     X_train_scaled = scaler.transform(split_data.X_train);
     X_test_scaled = scaler.transform(split_data.X_test); // Apply SAME transform to test data

     if(X_train_scaled.empty() || X_test_scaled.empty()) {
        cerr << "Error scaling data. Scaled data is empty." << endl;
        return 1;
     }
     cout << "Scaling complete." << endl;


    // 4. Train Logistic Regression Model (OvR from scratch)
    cout << "\nTraining Logistic Regression (OvR from scratch)..." << endl;
    LogisticRegressionOvR model(LEARNING_RATE, MAX_ITERATIONS);
    auto start_train = chrono::high_resolution_clock::now();
     model.fit(X_train_scaled, split_data.y_train, all_data.unique_classes);

    auto end_train = chrono::high_resolution_clock::now();
    chrono::duration<double> train_duration = end_train - start_train;
     cout << "Training finished in " << fixed << setprecision(2) << train_duration.count() << " seconds." << endl;


    // 5. Evaluate Model
    cout << "\nEvaluating model (scratch metrics)..." << endl;
    Labels y_train_pred, y_test_pred;
    y_train_pred = model.predict(X_train_scaled);
    y_test_pred = model.predict(X_test_scaled);

    // Check if predictions were successful
    if (y_train_pred.empty() || y_test_pred.empty()) {
         // predict method prints errors if model wasn't fitted or probabilities failed
         cerr << "Error during prediction. Predictions are empty." << endl;
         // Depending on which failed, print appropriate message
         if (y_train_pred.empty() && !X_train_scaled.empty()) cerr << "  Training set prediction failed." << endl;
         if (y_test_pred.empty() && !X_test_scaled.empty()) cerr << "  Testing set prediction failed." << endl;
         return 1;
    }


    // Calculate Accuracies
    float train_accuracy = calculateAccuracyScratch(split_data.y_train, y_train_pred);
    float test_accuracy = calculateAccuracyScratch(split_data.y_test, y_test_pred);

    cout << "\n--- Results for LogisticRegression (Scratch) ---" << endl;
    cout << fixed << setprecision(2);
    cout << "Training Accuracy: " << train_accuracy * 100.0f << "%" << endl;
    cout << "Testing Accuracy:  " << test_accuracy * 100.0f << "%" << endl;

    // Detailed Evaluation on TEST set
     evaluateModelScratch(split_data.y_test, y_test_pred, all_data.unique_classes);
    return 0;
}