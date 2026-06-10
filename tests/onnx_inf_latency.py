import onnxruntime as ort
import numpy as np
import time

model_path = "C:\\Users\\aiden\\OneDrive\\Desktop\\DRIFT\\DRIFT\\models\\drift_dinov2.onnx"

session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

print("Inputs:", [i.name for i in session.get_inputs()])
print("Outputs:", [(o.name, o.shape) for o in session.get_outputs()])
# # dummy data
# input_meta = session.get_inputs()[0]
# input_name = input_meta.name
# input_shape = input_meta.shape
# shape = [1 if isinstance(dim, str) or dim is None else dim for dim in input_shape]
# input_data = np.random.randn(*shape).astype(np.float32)

# # warm up
# session.run(None, {input_name: input_data})

# num_iterations = 100
# latencies = []

# for _ in range(num_iterations):
#     start_time = time.perf_counter()
#     session.run(None, {input_name: input_data})
#     latency = time.perf_counter() - start_time
#     latencies.append(latency * 1000) # Convert to milliseconds



# with open("output.txt", "a", encoding="utf-8") as file:
#     file.write("ONNX Inference Latency (CPU):\n")
#     file.write(f"Average Latency: {np.mean(latencies):.2f} ms\n")
#     file.write(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms\n")
#     file.write(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms\n")


