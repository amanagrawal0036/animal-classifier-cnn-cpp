#include <iostream>
#include <string>

#include "image_loader.h"
#include "cnn_layers.h"
#include "cnn.h"

#include <vector>
#include <map>

#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>

using namespace std;

int main() {
    // model configuration (used const bcs these should not change)
    const string csvPath = "image_data.csv";
    const int NUM_CLASSES = 3;
    const int IMAGE_HEIGHT = 150;
    const int IMAGE_WIDTH = 150;
    const int IMAGE_DEPTH = 1;  //This is for grayscaling images (for light computation)

    const int EPOCHS = 10;
    const float LEARNING_RATE = 0.001f;

    // Data Loading and Splitting Parameters (80:20 split)
    const int NUM_TRAIN_TARGET = 1200;
    const int NUM_TEST_TARGET = 300;
    const int LOAD_LIMIT = NUM_TRAIN_TARGET + NUM_TEST_TARGET;

    cout << "Loading image data from: " << csvPath << endl;
    auto csvData = readCSV(csvPath);

    if (csvData.empty()) {
        cerr << "Error: No data read from CSV file or file error. Exiting." << endl;
        return 1;
    }
    cout << "Found " << csvData.size() << " entries in CSV." << endl;
    cout << "Attempting to load up to " << LOAD_LIMIT << " images..." << endl;

    vector<ImageData> all_loaded_images;
    // reserving space of dataset (added this as encountered error when loading large dataset)
    all_loaded_images.reserve(LOAD_LIMIT);
    int loaded_count = 0;
    map<int, int> class_counts;

    for (const auto& [filepath, label] : csvData) {
        // Stop if we hit the defined limit
        if (loaded_count >= LOAD_LIMIT) {
            break;
        }
         if (label < 0 || label >= NUM_CLASSES) {
             cerr << "Warning: Skipping image " << filepath << " with invalid label " << label
                      << " (expected 0-" << NUM_CLASSES-1 << ")" << endl;
             continue;
         }

        ImageData img = loadImage(filepath, label);

        if (!img.data.empty()) {
            // cout<<"data size:"<<img.data.size()<<"\n";
            if ((int)img.data.size() != IMAGE_HEIGHT || (int)img.data[0].size() != IMAGE_WIDTH) {
                cout << "W1: image " << filepath << " dim not matching (" << img.data.size() << "x" << img.data[0].size() << ") after loading so skipped!!!" << endl;
                continue;
            }
            all_loaded_images.push_back(move(img));
            class_counts[label]++;
            loaded_count++;
        } else {
            cout << "failed to load image" << endl;
        }
    }
    cout<<"successfully loaded"<< all_loaded_images.size()<<" images." <<endl;

    if (all_loaded_images.empty()) {
        cerr << "Error: No images were loaded successfully. Exiting." << endl;
        return 1;
    }

    cout << "Class distribution in loaded data:" << endl;
    for(const auto& pair : class_counts) {
        cout << "  Label " << pair.first << ": " << pair.second << " images" << endl;
    }
     if ((int)class_counts.size() < NUM_CLASSES && !all_loaded_images.empty()) {
        cout << "Warning: data has less classes than defined for our config!!" << endl;
     } else if (class_counts.empty() && !all_loaded_images.empty()) {
         cout << "Warning: images dont have labels as expected by our config" << endl;
     }

     //Shuffles dataset for better outcome (so that model dont memorize the data)
    cout << "\nShuffling loaded images..." << endl;
    mt19937 rng(random_device{}());
    shuffle(all_loaded_images.begin(), all_loaded_images.end(), rng);

    vector<ImageData> training_data;// Keep separate training data
    vector<ImageData> testing_data;// Keep separate testing data
    int total_loaded = all_loaded_images.size();
    int num_train = 0;
    int num_test = 0;

    if (total_loaded >= (NUM_TRAIN_TARGET + NUM_TEST_TARGET)) {
        num_train = NUM_TRAIN_TARGET;
        num_test = NUM_TEST_TARGET;
        cout << "Sufficient data loaded. Splitting into " << num_train << " training and " << num_test << " testing samples." << endl;
    } else {
        num_train = static_cast<int>(total_loaded * 0.8);
        num_test = total_loaded - num_train;
        cout << "Warning: Loaded fewer images (" << total_loaded << ") than required for "
                 << NUM_TRAIN_TARGET << "/" << NUM_TEST_TARGET << " split." << endl;
        cout << "Using fallback 80/20 split: " << num_train << " training, " << num_test << " testing samples." << endl;
    }

    // reserving space of dataset (same reason as above)
    training_data.reserve(num_train);
    testing_data.reserve(num_test);

    cout << "movin data into training/testing sets " << endl;
    for (int i = 0; i < num_train; ++i) {
        if (i < (int)all_loaded_images.size()) {
             training_data.push_back(move(all_loaded_images[i]));
        }
    }
    for (int i = num_train; i < num_train + num_test; ++i) {
         if (i < (int)all_loaded_images.size()) {
             testing_data.push_back(move(all_loaded_images[i]));
         }
    }


     cout << "Training set size: " << training_data.size() << endl;
     cout << "Testing set size: " << testing_data.size() << endl;

    cout << "\n--- Building CNN Model ---" << endl;
    CNN cnn;

    // Architecture Definition
    // Layer 1: Conv -> Input (1, 150, 150), Output (8, 146, 146)
    cnn.addLayer<ConvLayer>(IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 8, 5, 1);
    
    // Layer 2: ReLU -> Input (8, 146, 146), Output (8, 146, 146)
    //cout<"Hello 2"<<"\n";
    cnn.addLayer<ReLULayer>(8, 146, 146);
    //cout<"Hello 3"<<"\n";

    // Layer 3: Pool -> Input (8, 146, 146), Output (8, 73, 73)
    cnn.addLayer<MaxPoolingLayer>(8, 146, 146, 2, 2);
    // Layer 4: Conv -> Input (8, 73, 73), Output (16, 71, 71)
    cnn.addLayer<ConvLayer>(8, 73, 73, 16, 3, 1);
     // Layer 5: ReLU -> Input (16, 71, 71), Output (16, 71, 71)

    //cout<"Hello 4"<<"\n";
    cnn.addLayer<ReLULayer>(16, 71, 71);
    //cout<"Hello 5"<<"\n";

    // Layer 6: Pool -> Input (16, 71, 71), Output (16, 35, 35)
    cnn.addLayer<MaxPoolingLayer>(16, 71, 71, 2, 2);
    // Layer 7: Flatten -> Input (16, 35, 35), Output (19600, 1, 1)
    const int flattened_size = 16 * 35 * 35;
    cnn.addLayer<FlattenLayer>(16, 35, 35);
    // Layer 8: Dense -> Input (19600), Output (128, 1, 1)
    cnn.addLayer<DenseLayer>(flattened_size, 128);
    // Layer 9: ReLU -> Input (128, 1, 1), Output (128, 1, 1)
    cnn.addLayer<ReLULayer>(128, 1, 1);
    // Layer 10: Dense -> Input (128), Output (NUM_CLASSES=6, 1, 1)
    cnn.addLayer<DenseLayer>(128, NUM_CLASSES);
    //cout<<"Hello 5"<<endl;
    // Layer 11: Softmax -> Input (NUM_CLASSES=6), Output (NUM_CLASSES=6, 1, 1)
    cnn.addLayer<SoftmaxLayer>(NUM_CLASSES);

    cout << "--- CNN Model Built Successfully ---" << endl;

    if (training_data.empty()) {
          cerr << "Error 2" << endl;
          return 1;
    }

    // Train using just the training_data
    cnn.train(training_data, NUM_CLASSES, EPOCHS, LEARNING_RATE);


    cout << "\n--- Evaluating on Training Data ---" << endl;
    int train_correct_predictions = 0;
    if (!training_data.empty()) {
        for (const auto& img : training_data) {
            if (img.data.empty()) {
                cerr << "Warning: Encountered empty image data during training evaluation." << endl;
                continue;
            }
            int prediction = cnn.predict(img);
            if (prediction == img.label) {
                train_correct_predictions++;
            }
        }
        double train_accuracy = (training_data.empty() ? 0.0 : static_cast<double>(train_correct_predictions) / training_data.size());
        cout << "Training Accuracy: " << train_accuracy * 100.0 << "% ("
                 << train_correct_predictions << "/" << training_data.size() << ")" << endl;
    } else {
         cout << "No training data to evaluate." << endl;
    }

    //evaluation on Testing Data
    cout << "\n--- Evaluating on Testing Data ---" << endl;
    int test_correct_predictions = 0;
    if (!testing_data.empty()) {
        for (const auto& img : testing_data) {
             if (img.data.empty()) {
                 cout << "Warning: img empty" << endl;
                 continue;
             }
             int prediction = cnn.predict(img);
             if (prediction == img.label) {
                 test_correct_predictions++;
             }
        }
        double test_accuracy = (testing_data.empty() ? 0.0 : static_cast<double>(test_correct_predictions) / testing_data.size());
        cout << "Testing Accuracy: " << test_accuracy * 100.0 << "% ("
                 << test_correct_predictions << "/" << testing_data.size() << ")" << endl;
    } else {
         cout << "No testing data to evaluate." << endl;
    }

      if (!testing_data.empty()) {
          cout << "\n--- Example Prediction (from Test Set) ---" << endl;
          const auto& first_test_image = testing_data[0];
          if (!first_test_image.data.empty()) {
               int pred_label = cnn.predict(first_test_image);
               cout << "Predicted label for the first test image: " << pred_label << endl;
               cout << "Actual label: " << first_test_image.label << endl;
          } else {
                 cout << "img empty" << endl;
          }
      }

    cout << "\n--- Program Finished ---" << endl;
    return 0;
}