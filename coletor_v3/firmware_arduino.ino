/*
 * FIRMWARE: Transdutor Capacitivo (Modo Escravo Serial com Reset de Oscilador)
 * Aplicação: Identificação de Metanol em Bebidas Alcoólicas
 * Descrição: Envia pings seriais "RT_TEMP:0.0" em estado ocioso para manter o handshake 
 * com o datalogger em Python. Ao receber 'I', aciona o ciclo High-Z estabilizado 
 * (com blanking de 1s via RESET do NE555) e devolve a matriz de CPJ estruturada.
 */

#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);

// --- MAPEAMENTO DE HARDWARE ---
const int pinoOscilador = 5; 
const int pinoReset555  = 7; // LIGAR AO PINO 4 DO NE555

// --- CHAVEAMENTO HIGH-Z (RESISTORES DE CARGA R1) ---
const int pinoR1_10k = 10;
const int pinoR1_4k7 = 9;
const int pinoR1_1k  = 8;

// --- CONSTANTES EXPERIMENTAIS ---
const int tempoAcomodacaoSegundos = 15; 
const int totalAmostrasJanela = 10; // 10 leituras de 100ms = 1 segundo cravado
float resultadosCPJ[5];                 

unsigned long tempoUltimaLeitura = 0;
const unsigned long intervaloLeitura = 1000; 

void setup() {
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();
  
  pinMode(pinoOscilador, INPUT);
  
  // Controle do Reset do 555
  pinMode(pinoReset555, OUTPUT);
  digitalWrite(pinoReset555, LOW); // Oscilador inicia desligado
  
  // Inicialização do Contador por Hardware (Timer1)
  TCCR1A = 0; 
  TCCR1B = 0;
  TCCR1B = (1 << CS12) | (1 << CS11) | (1 << CS10); 

  // Todos os resistores em alta impedância inicial
  colocarTodosEmHighZ();

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Aguardando PC...");
}

void colocarTodosEmHighZ() {
  pinMode(pinoR1_10k, INPUT);
  pinMode(pinoR1_4k7, INPUT);
  pinMode(pinoR1_1k, INPUT);
}

void configurarBancoResistores(int passo) {
  colocarTodosEmHighZ();
  switch (passo) {
    case 0: // 10k isolado
      pinMode(pinoR1_10k, OUTPUT); digitalWrite(pinoR1_10k, HIGH); break;
    case 1: // 4.7k isolado
      pinMode(pinoR1_4k7, OUTPUT); digitalWrite(pinoR1_4k7, HIGH); break;
    case 2: // 3.19k (10k e 4.7k em paralelo)
      pinMode(pinoR1_10k, OUTPUT); digitalWrite(pinoR1_10k, HIGH);
      pinMode(pinoR1_4k7, OUTPUT); digitalWrite(pinoR1_4k7, HIGH); break;
    case 3: // 1k isolado
      pinMode(pinoR1_1k, OUTPUT); digitalWrite(pinoR1_1k, HIGH); break;
    case 4: // 824R (1k e 4.7k em paralelo)
      pinMode(pinoR1_1k, OUTPUT); digitalWrite(pinoR1_1k, HIGH);
      pinMode(pinoR1_4k7, OUTPUT); digitalWrite(pinoR1_4k7, HIGH); break;
  }
}

void loop() {
  // Transmissão contínua de keep-alive em estado ocioso
  if (Serial.available() == 0) {
    unsigned long tempoAtual = millis();
    if (tempoAtual - tempoUltimaLeitura >= intervaloLeitura) {
      
      // Atualiza display local
      lcd.setCursor(0, 1); 
      lcd.print("Pronto P/ Coleta");
      
      // Envia ping vazio para o Python não travar
      Serial.println("RT_TEMP:0.0");
      
      tempoUltimaLeitura = tempoAtual;
    }
  }

  // Interrupção para execução da matriz de varredura
  if (Serial.available() > 0) {
    char comando = Serial.read();
    
    if (comando == 'I') {
      lcd.clear();
      Serial.println("Protocolo Iniciado...");
      
      // Acomodação inicial do fluido no sensor (Saquinho)
      for (int i = tempoAcomodacaoSegundos; i > 0; i--) {
        lcd.setCursor(0, 0); lcd.print("Acomodando Fluido");
        lcd.setCursor(0, 1); 
        lcd.print("Tempo: ");
        if(i < 10) lcd.print("0");
        lcd.print(i); lcd.print("s   ");
        delay(1000);
      }
      
      // Varredura Rigorosa com Reset Térmico/Elétrico
      for (int passo = 0; passo < 5; passo++) {
        
        // 1. Desliga o 555
        digitalWrite(pinoReset555, LOW);
        delay(10); 
        
        // 2. Chaveia a malha High-Z
        configurarBancoResistores(passo);
        
        // 3. Liga o 555
        digitalWrite(pinoReset555, HIGH);
        
        // 4. Acomodação elétrica obrigatória
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("Estabilizando R"); lcd.print(passo);
        delay(1000);
        
        // 5. Contagem de pulsos (Janela de 1s real)
        lcd.setCursor(0, 1); lcd.print("Lendo Freq...   ");
        unsigned long acumulador = 0;
        for (int am = 0; am < totalAmostrasJanela; am++) {
          TCNT1 = 0;
          delay(100);
          acumulador += TCNT1;
        }
        resultadosCPJ[passo] = acumulador / (float)totalAmostrasJanela;
      }
      
      // Desliga o 555 após concluir a varredura
      digitalWrite(pinoReset555, LOW);
      colocarTodosEmHighZ();
      
      // Envio do vetor consolidado para o Python
      // O '0.0' substitui a temperatura do LM35 para manter o CSV alinhado
      Serial.print("DATA:");
      Serial.print("0.0"); Serial.print(",");
      Serial.print(resultadosCPJ[0], 1); Serial.print(",");
      Serial.print(resultadosCPJ[1], 1); Serial.print(",");
      Serial.print(resultadosCPJ[2], 1); Serial.print(",");
      Serial.print(resultadosCPJ[3], 1); Serial.print(",");
      Serial.println(resultadosCPJ[4], 1);
      
      lcd.clear();
      lcd.setCursor(0, 0); lcd.print("Aguardando PC...");
      tempoUltimaLeitura = 0; 
    }
  }
}
