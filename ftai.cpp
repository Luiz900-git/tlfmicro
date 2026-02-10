#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/pwm.h"    // Biblioteca do PWM
#include "hardware/clocks.h" // Biblioteca de Clocks

// Certifique-se que o arquivo .h gerado no Python está na mesma pasta
#include "motor_model.h" 

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- CONFIGURAÇÃO DOS PINOS ---
#define LED_R_PIN 13
#define LED_G_PIN 11 
#define LED_B_PIN 12
#define JOYSTICK_X_PIN 26 // ADC0 - Vibração
#define JOYSTICK_Y_PIN 27 // ADC1 - Temperatura
#define BUZZER_PIN 21     // Pino do Buzzer

// Frequência do Buzzer (Sugestão: 100Hz é grave, se quiser mais agudo mude para 1000 ou 2000)
#define BUZZER_FREQUENCY 3000 

// Memória para o modelo (6KB)
const int kArenaSize = 6 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

// --- FUNÇÕES AUXILIARES ---

void set_led_color(bool r, bool g, bool b) {
    gpio_put(LED_R_PIN, r);
    gpio_put(LED_G_PIN, g);
    gpio_put(LED_B_PIN, b);
}

// Inicializa o PWM do Buzzer
void pwm_init_buzzer(uint pin) {
    gpio_set_function(pin, GPIO_FUNC_PWM);
    uint slice_num = pwm_gpio_to_slice_num(pin);
    pwm_config config = pwm_get_default_config();
    
    // Configura divisor de clock para atingir a frequência desejada
    pwm_config_set_clkdiv(&config, clock_get_hz(clk_sys) / (BUZZER_FREQUENCY * 4096)); 
    pwm_init(slice_num, &config, true);
    pwm_set_gpio_level(pin, 0); // Começa desligado
}

// Toca o Buzzer por um tempo determinado
void beep(uint pin, uint duration_ms) {
    uint slice_num = pwm_gpio_to_slice_num(pin);
    
    // Duty cycle de 50% (Som ligado)
    pwm_set_gpio_level(pin, 2048); 
    
    sleep_ms(duration_ms);
    
    // Desliga o som
    pwm_set_gpio_level(pin, 0);
    
    // Pequena pausa para não virar um som contínuo irritante
    sleep_ms(50); 
}

// --- MAIN ---
int main() {
    stdio_init_all();
    
    // Inicializa LEDs
    gpio_init(LED_R_PIN); gpio_set_dir(LED_R_PIN, GPIO_OUT);
    gpio_init(LED_G_PIN); gpio_set_dir(LED_G_PIN, GPIO_OUT);
    gpio_init(LED_B_PIN); gpio_set_dir(LED_B_PIN, GPIO_OUT);
    
    // Inicializa Buzzer
    pwm_init_buzzer(BUZZER_PIN);

    // Inicializa ADC
    adc_init();
    adc_gpio_init(JOYSTICK_X_PIN);
    adc_gpio_init(JOYSTICK_Y_PIN);

    // 1. Carregar Modelo TinyML
    const tflite::Model* model = tflite::GetModel(motor_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erro: Versão do modelo incompatível.\n");
        return 1;
    }

    // 2. Resolver Operações
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape();
    resolver.AddMul();     
    resolver.AddAdd();

    // 3. Criar Interpretador
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kArenaSize, nullptr
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Erro de alocação de memória.\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Sistema Iniciado! Monitorando...\n");

    while (true) {
        // --- A. LEITURA ---
        adc_select_input(0); 
        float raw_vib = (float)adc_read();
        float input_vib = raw_vib / 4095.0f;
        
        adc_select_input(1); 
        float raw_temp = (float)adc_read();
        float input_temp = raw_temp / 4095.0f;

        // --- B. INFERÊNCIA ---
        input->data.f[0] = input_vib;
        input->data.f[1] = input_temp;

        if (interpreter.Invoke() == kTfLiteOk) {
            float prob_normal = output->data.f[0];
            float prob_alerta = output->data.f[1];
            float prob_perigo = output->data.f[2];

            // --- C. ATUAÇÃO E PRINT ---
            
            // Monta uma string com os valores de entrada para o monitor
            // Ex: "V: 0.85 T: 0.90 | "
            printf("V: %.2f T: %.2f | ", input_vib, input_temp);

            if (prob_normal > prob_alerta && prob_normal > prob_perigo) {
                set_led_color(0, 1, 0); // Verde
                // Completa a linha: "Status: NORMAL (98.5%)"
                printf("Status: NORMAL (%.1f%%)\n", prob_normal * 100);
                
            } else if (prob_alerta > prob_normal && prob_alerta > prob_perigo) {
                set_led_color(0, 0, 1); // Azul
                printf("Status: ALERTA (%.1f%%)\n", prob_alerta * 100);
                
            } else {
                set_led_color(1, 0, 0); // Vermelho
                printf("Status: PERIGO !!! (%.1f%%)\n", prob_perigo * 100);
                
                // Toca o Buzzer se for perigo
                beep(BUZZER_PIN, 100); 
            }
            
        } else {
            printf("Erro na inferência\n");
        }

        // Delay para dar tempo de ler o texto no monitor
        sleep_ms(200); 
    }
    return 0;
}