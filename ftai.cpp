#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"

// Certifique-se que o arquivo .h gerado no Python está na mesma pasta
#include "motor_model.h" 

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Definição dos Pinos (Padrão BitDogLab)
#define LED_R_PIN 13
#define LED_G_PIN 11 // Na BitDogLab, verifique se G é 11 e B é 12 (ou vice-versa)
#define LED_B_PIN 12
#define JOYSTICK_X_PIN 26 // ADC0 - Vamos usar para Vibração
#define JOYSTICK_Y_PIN 27 // ADC1 - Vamos usar para Temperatura

// AJUSTE IMPORTANTE: Aumentamos a memória para caber o novo modelo mais inteligente
const int kArenaSize = 6 * 1024; // 6KB
alignas(16) uint8_t tensor_arena[kArenaSize];

void set_led_color(bool r, bool g, bool b) {
    gpio_put(LED_R_PIN, r);
    gpio_put(LED_G_PIN, g);
    gpio_put(LED_B_PIN, b);
}

int main() {
    stdio_init_all();
    
    // Inicializa LEDs
    gpio_init(LED_R_PIN); gpio_set_dir(LED_R_PIN, GPIO_OUT);
    gpio_init(LED_G_PIN); gpio_set_dir(LED_G_PIN, GPIO_OUT);
    gpio_init(LED_B_PIN); gpio_set_dir(LED_B_PIN, GPIO_OUT);
    
    // Inicializa ADC
    adc_init();
    adc_gpio_init(JOYSTICK_X_PIN);
    adc_gpio_init(JOYSTICK_Y_PIN);

    // 1. Carregar o Modelo
    const tflite::Model* model = tflite::GetModel(motor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erro: Versão do modelo incompatível.\n");
        return 1;
    }

    // 2. Resolver Operações (Adicionei mais algumas por segurança)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape(); // Importante para modelos Keras modernos
    resolver.AddMul();     // As vezes usado na normalização interna
    resolver.AddAdd();

    // 3. Criar Interpretador
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kArenaSize, nullptr
    );

    // 4. Alocar Memória
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Erro: Falha ao alocar tensores. Memória insuficiente?\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Sistema TinyML Iniciado! Ajuste os potenciômetros.\n");

    while (true) {
        // Leitura e Normalização (0 a 4095 -> 0.0 a 1.0)
        
        // Eixo X (ADC 0) -> Vibração
        adc_select_input(0); 
        float input_vib = (float)adc_read() / 4095.0f;
        
        // Eixo Y (ADC 1) -> Temperatura
        adc_select_input(1); 
        float input_temp = (float)adc_read() / 4095.0f;

        // Preenche o Tensor de Entrada
        // [0] deve ser Vibração e [1] Temperatura (conforme treinado no Python)
        input->data.f[0] = input_vib;
        input->data.f[1] = input_temp;

        // Executa a Inferência
        if (interpreter.Invoke() == kTfLiteOk) {
            float prob_normal = output->data.f[0];
            float prob_alerta = output->data.f[1];
            float prob_perigo = output->data.f[2];

            // Lógica de Decisão (Quem ganhou?)
            if (prob_normal > prob_alerta && prob_normal > prob_perigo) {
                set_led_color(0, 1, 0); // Verde
                printf("Status: NORMAL (%.2f)\n", prob_normal);
            } else if (prob_alerta > prob_normal && prob_alerta > prob_perigo) {
                set_led_color(0, 0, 1); // Azul
                printf("Status: ALERTA (%.2f)\n", prob_alerta);
            } else {
                set_led_color(1, 0, 0); // Vermelho
                printf("Status: PERIGO (%.2f)\n", prob_perigo);
            }
            
            // Exibe os dados brutos para você conferir no vídeo
            printf("  > Vibe: %.2f | Temp: %.2f\n", input_vib, input_temp);
            
        } else {
            printf("Erro na inferência\n");
        }

        sleep_ms(200); // 5 leituras por segundo
    }
    return 0;
}