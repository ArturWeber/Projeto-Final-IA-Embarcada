import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import time
import os

# ===== Caminhos =====
modelo_path = "./modelo.onnx"
modelo_fixed_path = "./modelo_fixed.onnx"
quant_model_path = "./modelo_quant.onnx"

# ===== 1. Carregar modelo =====
modelo = onnx.load(modelo_path, load_external_data=True)
print("Modelo carregado.")

# Checar external data
data_file = modelo_path + ".data"
print("External data:", "OK" if os.path.exists(data_file) else "NÃO ENCONTRADO")

# ===== 2. Converter external data para interno =====
# ONNX Runtime precisa disso para quantização
# OBS: Função oficial está em onnx.external_data_helper, mas se não existir
# podemos salvar todos os weights dentro do arquivo principal:
for tensor in modelo.graph.initializer:
    tensor.ClearField('external_data')  # remove referência a arquivo externo

onnx.save(modelo, modelo_fixed_path)
print(f"Modelo convertido para internal data salvo em: {modelo_fixed_path}")

# ===== 3. Quantização dinâmica INT8 =====
quantize_dynamic(
    model_input=modelo_fixed_path,
    model_output=quant_model_path,
    weight_type=QuantType.QInt8,
    per_channel=False  # mais seguro para modelos com external data
)
print(f"Modelo quantizado salvo em: {quant_model_path}")

# ===== 2. Extrair dimensão do input =====
input_tensor = modelo.graph.input[0]
input_name = input_tensor.name

shape = []
for dim in input_tensor.type.tensor_type.shape.dim:
    if dim.dim_value > 0:
        shape.append(dim.dim_value)
    else:
        shape.append(1)

print("Input name:", input_name)
print("Input shape:", shape)

# ===== 3. Criar sessão (APENAS UMA VEZ) =====
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(
    modelo_path,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

# ===== 4. Warmup =====
for _ in range(10):
    inp = np.random.randn(*shape).astype(np.float32)
    sess.run(None, {input_name: inp})

# ===== 5. Benchmark =====
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