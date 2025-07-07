# 🧠 CNN Image Classifier - Streamlit App

A comprehensive Streamlit web application that demonstrates Convolutional Neural Networks (CNNs) for image classification. This app allows users to train custom CNN models, upload images for classification, and visualize model performance.

## ✨ Features

- **Interactive Model Training**: Train CNN models with customizable parameters
- **Real-time Image Classification**: Upload and classify images instantly
- **Model Visualization**: View model architecture, training history, and performance metrics
- **Beautiful UI**: Modern, responsive design with intuitive navigation
- **Educational**: Perfect for learning about CNNs and deep learning

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
   ```bash
   cd "StreamLit AI app_01"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📱 How to Use

### 1. Home Page
- Overview of the application features
- Quick start guide
- Sample visualizations

### 2. Train Model
- Configure model parameters (number of classes, epochs)
- Define custom class names
- Train your CNN model with real-time progress tracking
- View training history and performance graphs

### 3. Make Predictions
- Upload images (JPG, JPEG, PNG formats)
- Get instant classification results
- View confidence scores for all classes
- Interactive prediction visualizations

### 4. Model Info
- Detailed model architecture summary
- Training metrics and performance statistics
- Class information and configuration details

## 🏗️ Model Architecture

The CNN model uses the following architecture:

```
├── Conv2D (32 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Flatten
├── Dropout (0.5)
├── Dense (512 units, ReLU)
└── Dense (num_classes, Softmax)
```

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation

### Key Features
- **Session State Management**: Maintains model and training state across page navigation
- **Real-time Training**: Live progress updates during model training
- **Image Preprocessing**: Automatic image resizing and normalization
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Graceful handling of various input scenarios

## 🎯 Use Cases

- **Educational**: Learn about CNN architecture and training
- **Prototyping**: Quick image classification model development
- **Demonstration**: Showcase deep learning capabilities
- **Research**: Experiment with different model configurations

## 🔧 Customization

You can easily customize the application:

1. **Model Architecture**: Modify the `create_cnn_model()` function
2. **UI Styling**: Update the CSS in the `st.markdown()` sections
3. **Data Loading**: Replace `generate_sample_data()` with real dataset loading
4. **Additional Features**: Add new pages or functionality

## 📊 Sample Workflow

1. **Start the app** and navigate to the "Train Model" page
2. **Configure** your model (e.g., 3 classes: "Cat", "Dog", "Bird")
3. **Train** the model and watch real-time progress
4. **Navigate** to "Make Predictions" page
5. **Upload** an image and get instant classification results
6. **View** detailed model information and training history

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving the UI/UX
- Optimizing model performance
- Adding more visualization options
- Enhancing documentation

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce the number of training epochs or use smaller images

3. **Port Already in Use**: Use a different port
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **TensorFlow Warnings**: These are usually harmless and can be ignored

### Performance Tips

- Use smaller image sizes for faster processing
- Reduce the number of epochs for quicker training
- Close other applications to free up memory
- Use GPU acceleration if available

---

**Built with ❤️ using Streamlit and TensorFlow**

*CNN Image Classifier - Making Deep Learning Accessible*