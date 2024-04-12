import subprocess
import sys
import tensorflow as tf

def save_model(model, save_path_h5, save_path_tf, save_path_onnx):
    # Save in TensorFlow's H5 format
    model.save(save_path_h5)
    print(f"Model saved in HDF5 format at {save_path_h5}")

    # Save in TensorFlow's SavedModel format
    model.save(save_path_tf, save_format="tf")
    print(f"Model saved in SavedModel format at {save_path_tf}")

    # Convert and save in ONNX format
    subprocess.run([
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", save_path_tf,
        "--output", save_path_onnx
    ], check=True)
    print(f"Model converted to ONNX format and saved at {save_path_onnx}")
