import onnxruntime as ort
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# ===== Função de benchmark =====
def benchmark_onnx(model_path, optimized=False, multicore=False, pruning=False, N=1000):
    sess_options = ort.SessionOptions()
    
    # Configurações de otimização
    if optimized:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    # Configurações de multithreading
    if multicore:
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = 1
    else:
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
    
    # Criar sessão
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    
    # Input
    input_tensor = sess.get_inputs()[0]
    input_name = input_tensor.name
    shape = [dim if isinstance(dim, int) else 1 for dim in input_tensor.shape]
    
    # Warmup
    for _ in range(10):
        inp = np.random.randn(*shape).astype(np.float32)
        sess.run(None, {input_name: inp})
    
    # Benchmark
    start = time.time()
    for _ in range(N):
        inp = np.random.randn(*shape).astype(np.float32)
        sess.run(None, {input_name: inp})
    end = time.time()
    
    tempo_total = end - start
    tempo_medio = tempo_total / N
    return tempo_medio*1000  # ms por inferência

# ===== Cenários =====
modelo_path = "./modelo.onnx"
N = 1000  # número de inferências para teste rápido

scenarios = {
    "Sem otimização": {"optimized": False, "multicore": False, "pruning": False},
    "Pruning": {"optimized": False, "multicore": False, "pruning": True},  # simulado
    "Multicore": {"optimized": False, "multicore": True, "pruning": False},
    "Multicore + Pruning": {"optimized": False, "multicore": True, "pruning": True}  # simulado
}

results = {}
for name, cfg in scenarios.items():
    print(f"Rodando: {name}")
    # Aqui pruning é apenas conceitual; você pode usar modelo já enxuto
    results[name] = benchmark_onnx(modelo_path, optimized=cfg["optimized"], multicore=cfg["multicore"], pruning=cfg["pruning"], N=N)

# ===== Plotar gráficos =====
labels = list(results.keys())
latencies = [results[label] for label in labels]

plt.figure(figsize=(10,5))
plt.bar(labels, latencies, color=["gray", "orange", "green", "blue"])
plt.ylabel("Latência média (ms)")
plt.title("Comparação de Latência por Configuração de Sessão ONNX Runtime")
plt.show()

# ===== Mostrar valores =====
for label in labels:
    print(f"{label}: {results[label]:.4f} ms")