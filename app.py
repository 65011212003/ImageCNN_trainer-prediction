import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Set page config
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 CNN Image Classifier</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Train Model", "Make Predictions", "Model Info"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Cat', 'Dog']  # Default classes

def create_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a CNN model for image classification"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def generate_sample_data():
    """Generate sample data for demonstration"""
    # Create synthetic data for demonstration
    x_train = np.random.random((100, 224, 224, 3))
    y_train = np.random.randint(0, 2, (100,))
    x_val = np.random.random((20, 224, 224, 3))
    y_val = np.random.randint(0, 2, (20,))
    return x_train, y_train, x_val, y_val

def process_uploaded_images(uploaded_files_by_class, target_size=(224, 224)):
    """Process uploaded images and create training data"""
    x_data = []
    y_data = []
    
    for class_idx, uploaded_files in enumerate(uploaded_files_by_class):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Load and preprocess image
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image = image.resize(target_size)
                    image_array = np.array(image) / 255.0
                    
                    x_data.append(image_array)
                    y_data.append(class_idx)
                except Exception as e:
                    st.error(f"Error processing image {uploaded_file.name}: {str(e)}")
    
    if len(x_data) == 0:
        return None, None, None, None
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Split into train and validation sets (80-20 split)
    # Handle small datasets gracefully
    if len(x_data) < 10:  # If less than 10 images total
        # Use all data for training and a small portion for validation
        split_idx = max(1, len(x_data) // 5)  # At least 1 for validation
        indices = np.random.permutation(len(x_data))
        
        val_indices = indices[:split_idx]
        train_indices = indices[split_idx:]
        
        x_train, x_val = x_data[train_indices], x_data[val_indices]
        y_train, y_val = y_data[train_indices], y_data[val_indices]
    else:
        # Use stratified split for larger datasets
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
            )
        except ValueError:
            # Fallback to simple split if stratification fails
            x_train, x_val, y_train, y_val = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42
            )
    
    return x_train, y_train, x_val, y_val

# Home Page
if page == "Home":
    st.markdown('<h2 class="sub-header">Welcome to CNN Image Classifier!</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 What is this app?</h3>
        <p>This is a Streamlit application that demonstrates Convolutional Neural Networks (CNNs) 
        for image classification. You can train your own model or use a pre-trained one to classify images.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🚀 Features</h3>
        <ul>
        <li>Train custom CNN models</li>
        <li>Upload and classify images</li>
        <li>View model architecture and performance</li>
        <li>Interactive visualizations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>📋 How to use</h3>
        <ol>
        <li><strong>Train Model:</strong> Create and train a CNN model</li>
        <li><strong>Make Predictions:</strong> Upload images for classification</li>
        <li><strong>Model Info:</strong> View model details and performance</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample image placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        sample_image = np.random.random((224, 224, 3))
        ax.imshow(sample_image)
        ax.set_title("Sample Image for Classification")
        ax.axis('off')
        st.pyplot(fig)

# Train Model Page
elif page == "Train Model":
    st.markdown('<h2 class="sub-header">🏋️ Train Your CNN Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model parameters
        num_classes = st.number_input("Number of Classes", min_value=2, max_value=10, value=2)
        epochs = st.slider("Training Epochs", min_value=1, max_value=1000, value=5)
        
        # Class names input
        st.subheader("Class Names & Training Images")
        class_names = []
        uploaded_files_by_class = []
        
        for i in range(num_classes):
            st.markdown(f"**Class {i+1}:**")
            col_name, col_upload = st.columns([1, 2])
            
            with col_name:
                class_name = st.text_input(f"Name", value=f"Class_{i+1}", key=f"class_name_{i}")
                class_names.append(class_name)
            
            with col_upload:
                uploaded_files = st.file_uploader(
                    f"Upload images for {class_name}",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key=f"upload_{i}",
                    help=f"Upload multiple images for the {class_name} class"
                )
                uploaded_files_by_class.append(uploaded_files)
                
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} images uploaded for {class_name}")
            
            st.markdown("---")
        
        st.session_state.class_names = class_names
        
        # Data source selection
        st.subheader("Training Data Source")
        use_uploaded_data = st.radio(
            "Choose training data source:",
            ["Use uploaded images", "Use synthetic data (demo)"],
            help="Select whether to use your uploaded images or synthetic data for demonstration"
        )
        
        # Check if we have uploaded data
        total_uploaded_images = sum(len(files) if files else 0 for files in uploaded_files_by_class)
        
        if use_uploaded_data == "Use uploaded images" and total_uploaded_images == 0:
            st.warning("⚠️ Please upload images for at least one class to use uploaded data.")
            st.info("💡 Tip: Upload at least 10-20 images per class for better training results.")
        
        # Training button
        can_train = True
        if use_uploaded_data == "Use uploaded images":
            can_train = total_uploaded_images >= num_classes * 2  # At least 2 images per class
            if not can_train:
                st.error("❌ Insufficient training data. Please upload at least 2 images per class.")
        
        if st.button("🚀 Start Training", type="primary", disabled=not can_train):
            with st.spinner("Training model... This may take a few minutes."):
                # Create model
                model = create_cnn_model(num_classes=num_classes)
                
                # Prepare training data based on selection
                if use_uploaded_data == "Use uploaded images":
                    st.info("📸 Processing uploaded images...")
                    x_train, y_train, x_val, y_val = process_uploaded_images(uploaded_files_by_class)
                    
                    if x_train is None:
                        st.error("❌ Failed to process uploaded images. Please check your files and try again.")
                        st.stop()
                    
                    st.success(f"✅ Processed {len(x_train)} training images and {len(x_val)} validation images")
                    
                    # Display data distribution
                    train_dist = np.bincount(y_train)
                    val_dist = np.bincount(y_val)
                    
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.write("**Training Data Distribution:**")
                        for i, (name, count) in enumerate(zip(class_names, train_dist)):
                            st.write(f"- {name}: {count} images")
                    
                    with col_dist2:
                        st.write("**Validation Data Distribution:**")
                        for i, (name, count) in enumerate(zip(class_names, val_dist)):
                            st.write(f"- {name}: {count} images")
                else:
                    st.info("🎲 Using synthetic data for demonstration...")
                    x_train, y_train, x_val, y_val = generate_sample_data()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback to update progress
                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f}')
                
                # Train model
                history = model.fit(
                    x_train, y_train,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[StreamlitCallback()],
                    verbose=0
                )
                
                # Save model to session state
                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.history = history
                st.session_state.training_data_info = {
                    'data_source': use_uploaded_data,
                    'total_train_images': len(x_train),
                    'total_val_images': len(x_val),
                    'num_classes': num_classes
                }
                
                st.success("✅ Model training completed!")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Model Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                
                st.pyplot(fig)
    
    with col2:
        st.subheader("Training Status")
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready!")
        else:
            st.info("ℹ️ No model trained yet")
        
        st.subheader("Model Architecture Preview")
        st.code("""
CNN Architecture:
├── Conv2D (32 filters, 3x3)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3)
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3)
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3)
├── MaxPooling2D (2x2)
├── Flatten
├── Dropout (0.5)
├── Dense (512 units, ReLU)
└── Dense (num_classes, Softmax)
        """)

# Make Predictions Page
elif page == "Make Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' page.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                help="Upload an image for classification"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Make prediction
                if st.button("🎯 Classify Image", type="primary"):
                    with st.spinner("Classifying image..."):
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        predictions = st.session_state.model.predict(processed_image)
                        predicted_class = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        
                        # Display results in col2
                        with col2:
                            st.subheader("Prediction Results")
                            
                            # Predicted class
                            st.metric(
                                "Predicted Class", 
                                st.session_state.class_names[predicted_class],
                                f"{confidence:.2%} confidence"
                            )
                            
                            # Confidence scores for all classes
                            st.subheader("Confidence Scores")
                            for i, (class_name, score) in enumerate(zip(st.session_state.class_names, predictions[0])):
                                st.write(f"**{class_name}:** {score:.2%}")
                                st.progress(float(score))
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(8, 6))
                            bars = ax.bar(st.session_state.class_names, predictions[0])
                            ax.set_title('Prediction Confidence by Class')
                            ax.set_ylabel('Confidence Score')
                            ax.set_ylim(0, 1)
                            
                            # Highlight predicted class
                            bars[predicted_class].set_color('red')
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
        
        with col2:
            if uploaded_file is None:
                st.info("👆 Upload an image to see predictions here")

# Model Info Page
elif page == "Model Info":
    st.markdown('<h2 class="sub-header">📊 Model Information</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' page.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Summary")
            
            # Create a string buffer to capture model summary
            import io
            buffer = io.StringIO()
            st.session_state.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            model_summary = buffer.getvalue()
            
            st.text(model_summary)
            
            st.subheader("Model Configuration")
            model_config = {
                "Total Parameters": st.session_state.model.count_params(),
                "Number of Layers": len(st.session_state.model.layers),
                "Input Shape": str(st.session_state.model.input_shape),
                "Output Shape": str(st.session_state.model.output_shape),
                "Optimizer": "Adam",
                "Loss Function": "Sparse Categorical Crossentropy"
            }
            
            # Add training data info if available
            if 'training_data_info' in st.session_state:
                training_info = st.session_state.training_data_info
                model_config.update({
                    "Data Source": training_info['data_source'],
                    "Training Images": training_info['total_train_images'],
                    "Validation Images": training_info['total_val_images'],
                    "Number of Classes": training_info['num_classes']
                })
            
            st.json(model_config)
        
        with col2:
            if 'history' in st.session_state:
                st.subheader("Training History")
                
                # Training metrics
                final_loss = st.session_state.history.history['loss'][-1]
                final_acc = st.session_state.history.history['accuracy'][-1]
                final_val_loss = st.session_state.history.history['val_loss'][-1]
                final_val_acc = st.session_state.history.history['val_accuracy'][-1]
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Final Training Loss", f"{final_loss:.4f}")
                    st.metric("Final Training Accuracy", f"{final_acc:.2%}")
                
                with col2_2:
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                    st.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")
                
                # Plot training curves
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Loss plot
                ax1.plot(st.session_state.history.history['loss'], 'b-', label='Training Loss')
                ax1.plot(st.session_state.history.history['val_loss'], 'r-', label='Validation Loss')
                ax1.set_title('Model Loss Over Time')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
                # Accuracy plot
                ax2.plot(st.session_state.history.history['accuracy'], 'b-', label='Training Accuracy')
                ax2.plot(st.session_state.history.history['val_accuracy'], 'r-', label='Validation Accuracy')
                ax2.set_title('Model Accuracy Over Time')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.subheader("Class Information")
            class_df = pd.DataFrame({
                'Class ID': range(len(st.session_state.class_names)),
                'Class Name': st.session_state.class_names
            })
            st.dataframe(class_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit and TensorFlow</p>
    <p>CNN Image Classifier - Deep Learning Made Simple</p>
</div>
""", unsafe_allow_html=True)