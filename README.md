# Projeto-Final-IA-Embarcada
Neural Network Warm-Start for Online Model Predictive Control (MPC). Final project for the Embedded AI course at USP. Accelerates solver convergence for real-time applications.


# README - Análise de Pré-Processamento e Loss para Problemas MPC em Controle de Quadrirrotor

Este diretório contém os códigos e notebooks de análise focados na metodologia de aprendizado de máquina para otimização de Problemas de Otimização Quadrática (QP) em contextos de Controle Preditivo Baseado em Modelo (MPC). O objetivo principal é desenvolver e avaliar uma Rede Neural (PlannerNet) para fornecer uma estimativa inicial (Warm Start) da variável de decisão ótima (primal, $z$) a um solucionador de primal active set online.

Este projeto replica e estende a metodologia apresentada em "Large Scale Model Predictive Control with Neural Networks and Primal Active Sets" e é aplicado ao controle preditivo de um quadrirrotor em ambiente de simulação para pouso em plataforma oscilante.

---

## 1. Configuração e Dependências

Os códigos dependem de bibliotecas comuns de cálculo científico e aprendizado profundo, incluindo **torch (PyTorch)**.

Para instalar as dependências, geralmente é utilizado um arquivo `requirements.txt`
O setup inicial define sementes (seeds) para reprodutibilidade usando **random**, **numpy** e **torch**.

---

## 2. Estrutura do Problema MPC e Pré-processamento

O código manipula instâncias de um problema MPC, que é um Problema de Otimização Quadrática (QP), carregado a partir de um arquivo `.npz` (`states_with_bounds.npz`).  
A classe **MPC_Problem** lida com o carregamento e verificação das dimensões do problema, incluindo:

- Número de estados (n_states) e atuações (n_actuations).
- Horizonte de predição (n_horizon).
- Dimensão da variável de decisão ($z$) (dim_z).
- Matrizes QP: $P$ (matriz quadrática), $q$ (vetor linear).
- Matrizes de Restrição: $C$ (igualdade), $U$ (desigualdade), e $D = [C; U]$.
- Variáveis Duais ($\lambda$, $\nu$) e limites de restrição ($z_{inf}$, $z_{sup}$).

### Pré-processamento Aplicado

Para melhorar a condicionalidade do problema e estabilizar o treinamento da rede, diversas transformações são aplicadas:

- **Padronização do Estado Inicial (State Standardization):**  
  $x' = (x - x_{mean}) / x_{std}$.

- **Padronização da Variável de Decisão (Primal Standardization):**  
  $z' = (z - z_{mean}) / z_{std}$, com propagação das transformações para $D$, $a$ e $b$.

- **Escalonamento dos Duais (Duals Scaling):**  
  Duais são normalizados pelo desvio padrão para equilibrar termos do Lagrangiano.

- **Regularização de Tikhonov:**  
  Adição de $\gamma I$ à matriz $P$ para garantir positividade definida e reduzir número de condição.

---

## 3. Arquitetura da Rede e Funções de Loss

### PlannerNet (Rede Neural)

A PlannerNet é uma MLP simples que recebe o estado padronizado ($x_{std}$) e retorna $z_{scaled}$.  
Usa ReLU como ativação e camadas ocultas (ex.: 64, 64 ou 128, 128), ajustadas via Grid Search.

### Funções de Loss

Foram definidas e validadas diferentes funções de perda:

- **LagrangianLoss** e **AugmentedLagrangianLoss:**  
  Baseadas na diferença entre Lagrangiano predito e ótimo.  
  A versão aumentada adiciona penalidade $\rho/2$ por violação das restrições.

- **QPLoss:**  
  Combina MSE do primal com penalização quadrática das violações.  
  $Loss = MSE(z_{pred}, z_{star}) + \lambda \cdot Penalty(\text{violações})$.

---

## 4. Treinamento e Análise de Estabilidade

O treinamento utiliza o dataset pré-processado e a função de perda escolhida.

- **Validação da Loss:** A perda deve ser próxima de zero quando $z_{pred} = z_{opt}$.
- **Grid Search:** Otimização de hiperparâmetros via validação cruzada K-Fold e Early Stopping.
- **Instabilidade de Gradiente:** O código monitora norma e direção do gradiente.  
  Explosões (> $10^3$) foram observadas em épocas específicas, geralmente devido ao termo quadrático da Lagrangiana.

---

## 5. Otimização e Implementação Embarcada (Deploy)

O modelo treinado é otimizado para execução em hardware embarcado, como **Raspberry Pi 4B** ou **Aquila AM69**.

### Formato de Deploy

- Exportação do modelo PyTorch para **ONNX**.
- Otimização via **ONNX Runtime**, com multithreading.

### Quantização

- **Quantização Pós-treino Dinâmica INT8** aplicada para reduzir tamanho do modelo.  
- Observou-se diminuição da latência (0.04173 ms → 0.03293 ms), porém aumento de MSE no Test Set.

### Arquitetura de Simulação

- Implementada em **C++** com ONNX Runtime multicore.
- A inclusão da NN (NN + qpOASES) aumentou iterações do solver, sugerindo piora na performance comparado ao uso isolado de qpOASES.

O estudo final foca na análise das funções de perda e integração com o solver.

---
