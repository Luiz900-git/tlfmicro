# 🏭 TinyML: Monitoramento Inteligente de Motores (Edge AI)

![Language](https://img.shields.io/badge/Language-C%2B%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20Pico-red)
![Framework](https://img.shields.io/badge/AI-TensorFlow%20Lite-orange)

Este projeto implementa um sistema de **Manutenção Preditiva** baseado em Inteligência Artificial (**TinyML**), rodando diretamente no microcontrolador RP2040 (BitDogLab).

O sistema é capaz de prever falhas em máquinas industriais analisando padrões de vibração e temperatura em tempo real, fornecendo feedback visual (LED RGB) e sonoro (Buzzer).

---

## 📋 Descrição Funcional

O projeto utiliza uma abordagem **HIL (Hardware-in-the-Loop)** para simulação:

1.  **Entrada de Dados (Sensores Simulados):**
    * **Eixo X do Joystick (ADC 0):** Simula o nível de **Vibração** da máquina.
    * **Eixo Y do Joystick (ADC 1):** Simula a **Temperatura** da carcaça.
2.  **Processamento (O Cérebro):**
    * Os dados são normalizados e alimentados em uma Rede Neural Artificial (Deep Learning) convertida para C++ via **TensorFlow Lite Micro**.
3.  **Saída (Atuadores):**
    * O sistema classifica o estado operacional e aciona os periféricos correspondentes.

## 🚦 Estados e Ações

A IA classifica o funcionamento da máquina em 3 categorias de risco:

| Probabilidade | Status | LED RGB | Buzzer (PWM) | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| **Classe 0** | ✅ **Normal** | **Verde** | *Desligado* | Operação segura e eficiente. |
| **Classe 1** | ⚠️ **Alerta** | **Azul** | *Desligado* | Sinais de desgaste. Manutenção preventiva sugerida. |
| **Classe 2** | 🚨 **Perigo** | **Vermelho** | **Ligado (Beep)** | Risco crítico de falha. Parada imediata recomendada. |

---

## 🛠️ Tecnologias Utilizadas

### Hardware
* **Placa:** BitDogLab (Raspberry Pi Pico / RP2040)
* **Sensores:** Joystick Analógico (2x Potenciômetros de 10kΩ)
* **Atuadores:** LED RGB e Buzzer Passivo

### Software & Ferramentas
* **Treinamento da IA:** Python, TensorFlow, Keras, Google Colab.
* **Firmware:** C++, Pico SDK, CMake.
* **Deploy:** TensorFlow Lite for Microcontrollers (TFLite).

---

## 📊 Pipeline de Desenvolvimento

1.  **Geração de Dataset:** Criação de dados sintéticos em Python simulando zonas de operação (Seguro, Alerta, Crítico).
2.  **Treinamento do Modelo:** Rede Neural Densa (Fully Connected) treinada para reconhecer padrões não-lineares.
3.  **Conversão:** O modelo treinado `.keras` foi convertido para um array de bytes C (`motor_model.h`) otimizado para memória flash.
4.  **Inferência:** O código C++ carrega o modelo e executa a classificação a cada 200ms.

---


