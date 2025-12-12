# Neural Network Warm-Start for MPC - Projeto de IA Embarcada

Este repositório contém a implementação de um sistema de controle híbrido para um veículo aéreo não tripulado (VANT/Drone). O projeto utiliza uma Rede Neural Profunda (DNN) para fornecer um "chute inicial" (*Warm Start*) a um controlador preditivo (MPC), visando acelerar a convergência e viabilizar a execução em hardware embarcado limitado (Raspberry Pi 4B).

## Estrutura do Repositório e Descrição dos Arquivos

Abaixo segue a explicação detalhada de cada arquivo presente neste projeto:

### 1\. Geração de Dados e Controle Clássico

  * **`mpc_explicit_controller.ipynb`**:
      * **Função:** Atua como o "Professor". Este notebook implementa a simulação dinâmica do drone e o controlador MPC clássico (usando solvers de otimização convexos como `cvxpy` ou `osqp`).
      * **Propósito:** Gera o *dataset* de treinamento (Ground Truth). Ele simula diversas trajetórias e salva os pares `(estado_atual, controle_otimo)` e `(estado_atual, primal_z)` que a rede neural tentará aprender.

### 2\. Desenvolvimento da IA

  * **`desenvolvModelo.ipynb`**:
      * **Função:** Atua como o "Aluno". É o notebook principal de *Deep Learning*.
      * **Propósito:**
        1.  Carrega e pré-processa os dados gerados.
        2.  Define a arquitetura da Rede Neural (MLP).
        3.  Implementa a função de perda customizada (`QPLoss`) que penaliza violações de restrição.
        4.  Treina o modelo e exporta o resultado final para o formato `.onnx`.
      * **Saída:** Gera os arquivos `modelo.onnx`.

### 3\. Modelos Treinados

  * **`modelo.onnx`**: O arquivo binário contendo a rede neural treinada exportada via PyTorch. É este arquivo que o sistema embarcado lê.
  * **`modelo_fixed.onnx`**: Uma versão do modelo pós-processada, geralmente onde foram aplicadas otimizações de grafo ou correções de metadados para compatibilidade com certas versões do *runtime*.
  * **`modelo.onnx.data`**: Arquivo auxiliar de pesos (geralmente gerado se o modelo for muito grande para um único arquivo protobuf, embora neste projeto sirva como artefato da exportação).

### 4\. Scripts de Execução e Teste (Python)

  * **`main.py`**:
      * **Função:** Script de automação geral.
      * **Propósito:** Serve como um *driver* para rodar inferências de teste ou integrar os módulos em Python. Pode ser usado para validar se o ambiente possui todas as dependências funcionando.
  * **`rodarMulticoreGraphopt.py`**:
      * **Função:** Benchmark específico de configurações do ONNX Runtime.
      * **Propósito:** Testa a inferência da rede variando o número de threads (Single Core vs Multi Core) e os níveis de otimização de grafo (Graph Optimization Level). Gera estatísticas de latência para análise de desempenho.

### 5\. Benchmarking em C (Hardware Nativo)

  * **`benchmark.c`**:
      * **Função:** Teste de estresse em baixo nível.
      * **Propósito:** Escrito em C puro, este código carrega o modelo ONNX usando a *C API* do ONNX Runtime. Ele é crucial para medir a latência real na Raspberry Pi, sem o *overhead* do interpretador Python. Utiliza `clock_gettime` para precisão de microssegundos.

### 6\. Configuração

  * **`requirements.txt`**: Lista de bibliotecas Python necessárias (PyTorch, ONNX, ONNX Runtime, NumPy, Matplotlib, etc.).

-----

## 🚀 Como Executar

### Pré-requisitos

Instale as dependências Python:

```bash
pip install -r requirements.txt
```

### Passo 1: Gerar Dados

Abra e execute todas as células do `mpc_explicit_controller.ipynb`. Isso criará os arquivos de dados (ex: `.csv` ou `.pt`) necessários para o treino.

### Passo 2: Treinar a Rede

Abra e execute o `desenvolvModelo.ipynb`. Certifique-se de que ele está apontando para os dados gerados no passo anterior. Ao final, ele salvará o arquivo `modelo.onnx`.

### Passo 3: Testar Inferência (Python)

Para verificar se o modelo roda corretamente e testar opções de otimização:

```bash
python rodarMulticoreGraphopt.py
```

### Passo 4: Benchmark em C (Linux/Raspberry Pi)

Para compilar o benchmark em C, você precisa ter o `libonnxruntime` instalado no sistema.

```bash
# Exemplo de compilação (ajuste os caminhos conforme sua instalação)
gcc benchmark.c -o benchmark -lonnxruntime

# Executar
./benchmark
```

-----

# 📄 Relatório do Projeto: Contexto e Resultados

*O texto abaixo descreve a motivação, metodologia e conclusões obtidas durante o desenvolvimento deste projeto na disciplina de Inteligência Artificial Embarcada.*

## 1\. Contexto e Motivação

### Contexto do Projeto

O presente trabalho foi desenvolvido no âmbito da disciplina de Inteligência Artificial Embarcada, visando a aplicação prática de técnicas de aprendizado profundo (*Deep Learning*) em sistemas de controle. O cenário de aplicação escolhido baseia-se em um problema real de robótica aérea: o pouso autônomo de um veículo aéreo não tripulado (VANT), especificamente um quadrotor, em condições adversas, como o pouso em plataformas móveis ou em ambientes marítimos (*automar*).

O cenário deriva de uma pesquisa onde a estratégia adotada foi o Controle Preditivo Baseado em Modelo (MPC). O MPC atua como o "cérebro" da aeronave, calculando a cada instante a sequência de ações ótimas. A técnica formula o controle como um problema de otimização matemática (Programação Quadrática - QP). Embora robusto, o MPC é computacionalmente oneroso, especialmente para hardware embarcado.

### Motivação e Desafios de Tempo Real

A principal motivação reside no custo computacional proibitivo do MPC. Em robótica aérea, o requisito de tempo real é crítico. Se o *solver* não entregar uma resposta a tempo, o drone pode cair.

A abordagem do projeto é utilizar uma **Rede Neural Profunda** para fornecer um *Warm Start* (partida quente) ao *solver*. A hipótese é que a rede, tendo tempo de inferência fixo e determinístico, pode entregar uma solução muito próxima da ótima, reduzindo drasticamente o número de iterações que o *solver* precisa para refinar o resultado.

## 2\. Metodologia e Adaptações

### Visão Geral da Abordagem Híbrida

Baseado no artigo *"Large Scale Model Predictive Control with Neural Networks and Primal Active Sets"*, o projeto combina uma rede neural (treinada offline) com um solucionador *Active Set* (online). A rede mapeia o estado atual ($x$) para uma aproximação das variáveis de otimização ($z$). O solver então utiliza esse $z$ como ponto de partida para garantir a viabilidade e otimalidade finais.

### Adaptações

Para a prova de conceito, simplificamos a abordagem original. Em vez de reescrever um solver QP do zero, acoplamos a rede neural a um solver padrão, focando na eficiência da inferência da rede no hardware alvo (Raspberry Pi 4B) e na validação do *Warm Start*.

## 3\. Método

Durante o desenvolvimento, a versão inicial do método (baseada em perda Lagrangiana complexa) não convergiu adequadamente. Adotou-se então uma abordagem simplificada e robusta.

### Geração de Dados

O problema foi formulado como *Box-constrained Quadratic Programming*. Utilizou-se uma arquitetura SIL (Software-in-the-Loop) para simular o drone, resolver o MPC clássico e coletar dados de: Estado inicial ($x$), Primal ótimo ($z^*$), e limites de restrição.

### Pré-processamento

  * **Normalização:** Aplicou-se *z-score* (média 0, desvio padrão 1) nos dados de entrada para facilitar o treinamento da rede.
  * **Condicionalidade:** Na versão inicial, tentou-se regularização de Tikhonov e normalização espectral, mas a versão final simplificou o processo focando na normalização dos estados.

### Função de Perda (Loss Function)

A abordagem inicial (**Lagrangian Loss**) falhou; a rede aprendia o valor escalar do custo, mas não o vetor de controle correto.

Desenvolvemos a **QPLoss** (implementada em `desenvolvModelo.ipynb`), que combina:

1.  **MSE (Erro Quadrático Médio):** Força a rede a imitar o controle ótimo ($z^*$).
2.  **Penalidade de Restrição:** Adiciona um custo proporcional à violação das restrições físicas ($Ax \le b$), agindo como uma *soft constraint*.

<!-- end list -->

```python
# Conceito da QPLoss
Loss = ||z_pred - z_star||^2 + lambda * sum(max(0, violação))
```

### Arquitetura da Rede e Otimização

Utilizou-se uma **MLP (Multilayer Perceptron)** rasa com ativação **ReLU**, ideal para aproximar as funções lineares por partes do MPC explícito.

  * **Modelo Vencedor:** 1 camada oculta com 128 neurônios.
  * **Resultados de Treino:** MSE de 0.1027 no conjunto de teste. Histograma de erros concentrado em zero.

## 4\. Otimização Computacional e Hardware

### Quantização (INT8)

Tentou-se quantização dinâmica via ONNX Runtime.

  * **Tamanho:** Redução de 250 KB para 50 KB.
  * **Latência:** Ganho marginal (\~10 $\mu$s).
  * **Precisão:** O erro MSE triplicou.
  * **Conclusão:** Não valeu a pena para este caso, pois prejudicou a qualidade do *Warm Start*.

### Multicore vs Single Core

Testes realizados com `benchmark.c` e `rodarMulticoreGraphopt.py`.

  * Para inferências muito rápidas (\~40 $\mu$s), o overhead de paralelização supera o ganho. A execução **Single Core** mostrou-se mais eficiente.

## 5\. Deploy e Validação em Malha Fechada

### Arquitetura de Software

O sistema embarcado foi desenvolvido em C++, integrando:

1.  **Motor de Inferência:** ONNX Runtime carregando `modelo.onnx`.
2.  **Solver MPC:** Recebe a saída da rede como inicialização.
3.  **Simulação Dinâmica:** Valida a física do drone.

### Resultados do Deploy

1.  **Validação Funcional (Sucesso):** O drone controlado pela Rede Neural + Solver realizou a trajetória de pouso perfeitamente, sobrepondo-se à curva do controle clássico. O sistema é seguro e funcional.
2.  **Performance Computacional (Desafio):**
      * O número de iterações do solver com *Warm Start* (Rede) foi **maior** (\~400 iterações) do que com *Cold Start* (\~50 iterações).
      * **Diagnóstico:** Embora a rede tenha um MSE baixo (visualmente correto), a solução é "numericamente rugosa". O solver gasta mais tempo projetando a solução "quase ótima" da rede de volta para a viabilidade estrita do que começando do zero.

### Conclusão

O projeto demonstrou com sucesso a viabilidade técnica de rodar IA embarcada para controle complexo na Raspberry Pi. A arquitetura híbrida funciona e controla o drone. O desafio remanescente é refinar a função de perda (voltando à teoria Lagrangiana rigorosa) para alinhar os gradientes da rede com os do solver, transformando a precisão visual em aceleração numérica real.
