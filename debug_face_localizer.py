import numpy as np
from PIL import Image
import tensorflow as tf

model_path = "artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite"
image_path = "/Users/hugocornellier/Downloads/cream-long-haired-dachshund-outside_Valeria-Head_Shutterstock.jpg"

# Load model
interp = tf.lite.Interpreter(model_path=model_path)
interp.allocate_tensors()
input_details = interp.get_input_details()
output_details = interp.get_output_details()

print("=== Input details ===")
print(input_details)
print("\n=== Output details ===")
print(output_details)

# Load and preprocess image
img = Image.open(image_path).convert("RGB")
print(f"\nOriginal size (W x H): {img.size}")

# Letterbox
w, h = img.size
scale = min(224/w, 224/h)
new_w, new_h = int(round(w*scale)), int(round(h*scale))
pad_x = (224 - new_w) // 2
pad_y = (224 - new_h) // 2
print(f"scale={scale}, new_w={new_w}, new_h={new_h}, pad_x={pad_x}, pad_y={pad_y}")

resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
canvas = Image.new("RGB", (224, 224), (0, 0, 0))
canvas.paste(resized, (pad_x, pad_y))
arr = np.asarray(canvas).astype(np.float32) / 255.0
print(f"Input array range: [{arr.min():.4f}, {arr.max():.4f}], shape: {arr.shape}, dtype: {arr.dtype}")

# Check a corner pixel (should be black padding)
print(f"Top-left pixel (should be black pad): {arr[0, 0, :]}")
# Check the center pixel
print(f"Center pixel: {arr[112, 112, :]}")

# Check corner pixels of PIL image array layout
print(f"arr[pad_y, pad_x] (first content pixel top-left): {arr[pad_y, pad_x, :]}")

# Run
inp = arr[np.newaxis].astype(input_details[0]['dtype'])
print(f"\nInput tensor shape sent: {inp.shape}, dtype: {inp.dtype}")
interp.set_tensor(input_details[0]['index'], inp)
interp.invoke()
output = interp.get_tensor(output_details[0]['index'])
print(f"\n=== RAW output (before any clip/sort) ===")
print(f"Shape: {output.shape}")
print(f"Values: {output}")

# Standard postprocessing
raw = output[0].astype(np.float32)
x1_n, y1_n, x2_n, y2_n = np.clip(raw, 0, 1)
print(f"\nClipped normalized [x1,y1,x2,y2]: [{x1_n:.6f}, {y1_n:.6f}, {x2_n:.6f}, {y2_n:.6f}]")

# Scale to letterbox pixel coords
x1 = x1_n * 224
y1 = y1_n * 224
x2 = x2_n * 224
y2 = y2_n * 224
print(f"In letterbox pixels [x1,y1,x2,y2]: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")

# De-letterbox
x1 -= pad_x; x2 -= pad_x
y1 -= pad_y; y2 -= pad_y
x1 /= scale; x2 /= scale
y1 /= scale; y2 /= scale
print(f"\nDe-letterboxed bbox (original image px): [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
print(f"Image size: {w}x{h}")
print(f"BBox width: {x2-x1:.1f}px, height: {y2-y1:.1f}px")
print(f"Is degenerate (w<1 or h<1)? {(x2-x1 < 1.0) or (y2-y1 < 1.0)}")
