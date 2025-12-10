#include <iostream>
#include <vector>
#include <random>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <map>
#include <string>
#include <thread>

// ===== Função de benchmark =====
double run_benchmark(const char* model_path, bool optimized, bool multicore, int N = 100000) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_example");
    Ort::SessionOptions session_options;

    // Configurações de otimização
    session_options.SetGraphOptimizationLevel(
        optimized ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED 
                  : GraphOptimizationLevel::ORT_DISABLE_ALL
    );

    // Configurações de multithreading
    session_options.SetIntraOpNumThreads(multicore ? std::thread::hardware_concurrency() : 1);
    session_options.SetInterOpNumThreads(1);

    // Criar sessão
    Ort::Session session(env, model_path, session_options);

    // Inputs
    std::vector<std::string> input_names_str = session.GetInputNames();
    if (input_names_str.empty()) {
        std::cerr << "ERRO: modelo não tem inputs!" << std::endl;
        return -1.0;
    }
    std::vector<const char*> input_names;
    for (auto& s : input_names_str) input_names.push_back(s.c_str());

    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();
    for (auto& s : input_shape) if (s < 0) s = 1;

    size_t total_elems = 1;
    for (auto s : input_shape) total_elems *= s;

    std::vector<float> input_tensor_values(total_elems);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    // Outputs
    std::vector<std::string> output_names_str = session.GetOutputNames();
    std::vector<const char*> output_names;
    for (auto& s : output_names_str) output_names.push_back(s.c_str());

    // Warmup
    for (int i = 0; i < 10; i++) {
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                    output_names.data(), output_names.size());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (auto& v : input_tensor_values) v = dist(gen);  // atualizar dados aleatórios
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                    output_names.data(), output_names.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double total_time = diff.count();
    double avg_time_ms = (total_time / N) * 1000.0;

    return avg_time_ms;
}

int main() {
    const char* model_path = "modelo.onnx";
    int N = 100000;

    // Cenários
    std::map<std::string, std::pair<bool,bool>> scenarios = {
        {"Single-core, não otimizado", {false, false}},
        {"Single-core, otimizado",   {true,  false}},
        {"Multicore, não otimizado", {false, true}},
        {"Multicore, otimizado",     {true,  true}}
    };

    std::map<std::string,double> results;

    for (auto& kv : scenarios) {
        std::cout << "Rodando: " << kv.first << std::endl;
        bool optimized = kv.second.first;
        bool multicore = kv.second.second;

        double avg_latency = run_benchmark(model_path, optimized, multicore, N);
        results[kv.first] = avg_latency;
        std::cout << "Tempo médio por inferência: " << avg_latency << " ms\n\n";
    }

    std::cout << "===== RESUMO =====\n";
    for (auto& kv : results)
        std::cout << kv.first << ": " << kv.second << " ms\n";

    return 0;
}