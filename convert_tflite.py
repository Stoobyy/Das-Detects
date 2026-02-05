"""Convert Keras model to TFLite"""
import tensorflow as tf
import sys

def convert_to_tflite(keras_path, tflite_path=None, quantize=False):
    """Convert .keras model to .tflite"""
    if tflite_path is None:
        tflite_path = keras_path.replace('.keras', '.tflite')
    
    # Load model
    print(f"Loading {keras_path}...")
    model = tf.keras.models.load_model(keras_path)
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Optional: Quantize for smaller size and faster inference
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying quantization...")
    
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Size comparison
    import os
    keras_size = os.path.getsize(keras_path) / 1024 / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024 / 1024
    
    print(f"\n✅ Converted successfully!")
    print(f"   Keras:  {keras_size:.2f} MB")
    print(f"   TFLite: {tflite_size:.2f} MB")
    print(f"   Saved:  {tflite_path}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "voice_classifier_model.keras"
    quantize = "--quantize" in sys.argv
    convert_to_tflite(model, quantize=quantize)
