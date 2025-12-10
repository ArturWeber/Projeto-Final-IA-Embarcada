import onnxruntime as ort
import numpy as np
import time
import os

# ===== Caminho do modelo =====
modelo_path = "./modelo.onnx"

# ===== Criar sessão otimizada =====
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.inter_op_num_threads = 1

sess = ort.InferenceSession(
    modelo_path,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

# ===== Extrair input info =====
input_tensor = sess.get_inputs()[0]
input_name = input_tensor.name
shape = input_tensor.shape
shape = [dim if isinstance(dim, int) else 1 for dim in shape]

print("Input name:", input_name)
print("Input shape:", shape)

# ===== Warmup =====
for _ in range(10):
    inp = np.random.randn(*shape).astype(np.float32)
    sess.run(None, {input_name: inp})

# ===== Benchmark =====
N = 3000000
start = time.time()
for _ in range(N):
    inp = np.random.randn(*shape).astype(np.float32)
    sess.run(None, {input_name: inp})
end = time.time()

tempo_total = end - start
tempo_medio = tempo_total / N

print(f"\n===== RESULTADOS =====")
print(f"Execuções: {N}")
print(f"Tempo total: {tempo_total:.4f} s")
print(f"Tempo médio por inferência: {tempo_medio*1000:.3f} ms")