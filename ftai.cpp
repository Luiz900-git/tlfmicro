#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"

#include "motor_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Definição dos Pinos (Padrão BitDogLab)
#define LED_R_PIN 13
#define LED_G_PIN 11
#define LED_B_PIN 12
#define JOYSTICK_X_PIN 26 // ADC0
#define JOYSTICK_Y_PIN 27 // ADC1

// Tamanho da memória reservada para o modelo rodar
const int kArenaSize = 2048;
alignas(16) uint8_t tensor_arena[kArenaSize];

// Função auxiliar para controlar o LED RGB
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
    
    // Inicializa ADC (Joystick/Potenciometros)
    adc_init();
    adc_gpio_init(JOYSTICK_X_PIN);
    adc_gpio_init(JOYSTICK_Y_PIN);

    // 1. Carregar o Modelo
    const tflite::Model* model = tflite::GetModel(motor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erro: Versão do modelo incorreta (%d != %d).\n", model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    // 2. Definir as Operações (Resolver)
    // Operações mais comuns para redes neurais simples
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape();

    // 3. Criar o Interpretador
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kArenaSize, nullptr
    );

    // 4. Alocar Memória para os Tensores
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Erro: Falha ao alocar tensores (Aumente kArenaSize se necessário).\n");
        return 1;
    }

    // Ponteiros para entrada e saída do modelo
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Sistema Iniciado! Lendo sensores...\n");

    while (true) {
        // Leitura dos Sensores (Simulando Vibração e Temperatura)
        // ADC 1 (Pino 27) -> Temperatura
        adc_select_input(1); 
        float input_temp = (float)adc_read() / 4095.0f;
        
        // ADC 0 (Pino 26) -> Vibração
        adc_select_input(0); 
        float input_vib = (float)adc_read() / 4095.0f;

        // Preenche o tensor de entrada
        input->data.f[0] = input_vib;
        input->data.f[1] = input_temp;

        // Roda a inferência (Classificação)
        if (interpreter.Invoke() == kTfLiteOk) {
            // Pega os resultados (Probabilidades)
            float normal = output->data.f[0];
            float alerta = output->data.f[1];
            float perigo = output->data.f[2];

            // Lógica para acender o LED baseado na maior probabilidade
            if (normal > alerta && normal > perigo) {
                set_led_color(0, 1, 0); // Verde (Normal)
            } else if (alerta > normal && alerta > perigo) {
                set_led_color(0, 0, 1); // Azul (Alerta)
            } else {
                set_led_color(1, 0, 0); // Vermelho (Perigo)
            }
            
            // Debug no Serial Monitor
            printf("Vib: %.2f | Temp: %.2f || Norm: %.2f Alert: %.2f Perig: %.2f\n", 
                   input_vib, input_temp, normal, alerta, perigo);
        } else {
            printf("Erro ao rodar inferência.\n");
        }

        sleep_ms(100); // Aguarda um pouco antes da próxima leitura
    }
    return 0;
}