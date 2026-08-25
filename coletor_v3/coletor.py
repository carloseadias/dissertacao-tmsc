"""
SOFTWARE DATALOGGER: Interface de Aquisição de Dados do Sensor Capacitivo
Descrição: Monitora continuamente a porta de comunicação, comunicando-se com o microcontrolador 
para buscar informações em tempo real e registrar propriedades multiparamétricas de amostras líquidas.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import serial
import time
import threading
from datetime import datetime
import winsound

# Constantes de configuração do equipamento
PORTA_SERIAL = 'COM4' 
TAXA_TRANSMISSAO = 9600
ARQUIVO_CSV = 'ds_cachaca_metanol.csv'

# =========================================================================================
# ESTRUTURA DE DADOS DAS AMOSTRAS (VETOR ASSOCIATIVO)
# Esta estrutura funciona de forma análoga a matrizes ou dicionários em outras linguagens.
# Para cadastrar uma nova bebida, adicione uma nova linha respeitando a formatação abaixo.
# =========================================================================================
DADOS_CACHACAS = {
    "Sagatiba Lote LJ/NH 07 Jun/19": {"teor_rotulo": 38, "teor_medido": 41, "nivel_acucar": 0},
    "Pirassununga 51 Lote T114:22 1 L 080426": {"teor_rotulo": 39, "teor_medido": 38, "nivel_acucar": 5},
    "Velho Barreiro Lote C1 22:18 F 22/01/26": {"teor_rotulo": 39, "teor_medido": 34, "nivel_acucar": 15},
    "Trivisan": {"teor_rotulo": 42, "teor_medido": 39, "nivel_acucar": 0}
}

class AplicativoColetor:
    def __init__(self, janela_raiz):
        self.raiz = janela_raiz
        self.raiz.title("Coletor - Dados do sensor capacitivo de cachaça e metanol")
        self.raiz.geometry("450x700") # Aumentado para acomodar o novo campo
        self.raiz.configure(padx=20, pady=20)
        
        self.conexao_serial = None
        self.conectar_equipamento()

        # Variáveis de controle para o laço de múltiplas coletas e séries
        self.leituras_restantes = 0
        self.series_restantes = 0
        self.leituras_por_serie = 0

        # =========================================================================================
        # CONSTRUÇÃO DA INTERFACE GRÁFICA DE USUÁRIO
        # =========================================================================================
        tk.Label(janela_raiz, text="METADADOS DA AMOSTRA", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        # Campo: Marca da Bebida (Caixa de seleção suspensa)
        tk.Label(janela_raiz, text="Marca da Bebida:").pack(anchor="w")
        self.selecao_marca = ttk.Combobox(janela_raiz, values=list(DADOS_CACHACAS.keys()), width=37, state="readonly")
        self.selecao_marca.pack(pady=(0, 10))
        self.selecao_marca.current(0) # Define a primeira bebida da lista como padrão

        # Campo: Grau de Pureza (Caixa de seleção suspensa)
        tk.Label(janela_raiz, text="Grau de Pureza (%):").pack(anchor="w")
        valores_pureza = ["100.0", "98.0", "95.0", "90.0", "85.0", "75.0", "70.0"]
        self.selecao_pureza = ttk.Combobox(janela_raiz, values=valores_pureza, width=37, state="readonly")
        self.selecao_pureza.pack(pady=(0, 10))
        self.selecao_pureza.current(0) # Define 100.0 como padrão

        # Campo: Temperatura Externa (Entrada de texto livre)
        tk.Label(janela_raiz, text="Temperatura Sensor Externo (°C):", font=("Arial", 10, "bold"), fg="#1A237E").pack(anchor="w")
        self.entrada_temp_ext = tk.Entry(janela_raiz, width=40, bg="#E8EAF6")
        self.entrada_temp_ext.pack(pady=(0, 10))
        self.entrada_temp_ext.insert(0, "25.0")
        
        # Campo: Quantidade de Leituras (Entrada de texto livre para definir o ciclo de repetição)
        tk.Label(janela_raiz, text="Quantidade de Leituras:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entrada_leituras = tk.Entry(janela_raiz, width=10)
        self.entrada_leituras.pack(anchor="w", pady=(0, 10))
        self.entrada_leituras.insert(0, "1")

        # Campo: Quantidade de Séries (Entrada de texto livre para definir repetições do ciclo)
        tk.Label(janela_raiz, text="Quantidade de Séries:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entrada_series = tk.Entry(janela_raiz, width=10)
        self.entrada_series.pack(anchor="w", pady=(0, 15))
        self.entrada_series.insert(0, "1")

        self.temp_externa_atual = "25.0"

        # Painel: Monitoramento do estado da placa e do sensor
        self.painel_status = tk.Frame(janela_raiz, bd=1, relief="solid", padx=10, pady=5)
        self.painel_status.pack(fill="x", pady=(0, 15))
        
        self.rotulo_status_titulo = tk.Label(self.painel_status, text="Status Hardware:", font=("Arial", 10))
        self.rotulo_status_titulo.pack(side="left")
        
        self.rotulo_status_valor = tk.Label(self.painel_status, text="Aguardando...", font=("Arial", 11, "bold"))
        self.rotulo_status_valor.pack(side="right")

        # Botão: Iniciar o processo de comunicação e aquisição
        self.botao_iniciar = tk.Button(janela_raiz, text="▶ INICIAR COLETA", font=("Arial", 12, "bold"), 
                                     bg="#4CAF50", fg="white", height=2, command=self.iniciar_processo_coleta)
        self.botao_iniciar.pack(fill="x", pady=(0, 15))

        # Painel: Console para exibir os registros de atividade em texto
        tk.Label(janela_raiz, text="Console de Comunicação:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.console_texto = tk.Text(janela_raiz, height=7, width=50, bg="#f4f4f4")
        self.console_texto.pack(fill="both", expand=True)

        self.preparar_arquivo_dados()
        
        # Inicializa a linha de execução paralela para monitorar a porta de comunicação continuamente
        self.execucao_ativa = True
        self.processo_leitura = threading.Thread(target=self.monitorar_porta_continuamente, daemon=True)
        self.processo_leitura.start()

    def registrar_atividade(self, mensagem):
        """Imprime mensagens no console da interface para acompanhamento do usuário."""
        self.console_texto.insert(tk.END, mensagem + "\n")
        self.console_texto.see(tk.END) 
        self.raiz.update()

    def conectar_equipamento(self):
        """Estabelece o canal de comunicação físico com o microcontrolador."""
        try:
            self.conexao_serial = serial.Serial(PORTA_SERIAL, TAXA_TRANSMISSAO, timeout=1)
            time.sleep(2) # Aguarda o reinício padrão da placa após conexão
        except Exception as e:
            messagebox.showerror("Falha de Hardware", f"Falha ao abrir {PORTA_SERIAL}.")
            self.raiz.destroy()

    def preparar_arquivo_dados(self):
        """Garante que o arquivo de armazenamento exista e escreve o cabeçalho caso esteja vazio."""
        try:
            with open(ARQUIVO_CSV, 'a', encoding='utf-8') as arquivo:
                if arquivo.tell() == 0:
                    # Cabeçalho ajustado para contemplar o NivelAcucar no lugar da variável binária anterior
                    arquivo.write("DataHora;Marca;TeorRotulo;TeorDensimetro;NivelAcucar;Temperatura;CPJ_10k;CPJ_4k7;CPJ_3k19;CPJ_1k;CPJ_824R;NivelPureza\n")
        except Exception as erro:
            self.registrar_atividade(f"[ERRO] Falha no arquivo CSV: {erro}")

    def iniciar_processo_coleta(self):
        """Captura os parâmetros da interface, valida os dados e dá início ao ciclo de requisições."""
        self.marca_atual = self.selecao_marca.get()
        
        # O cruzamento de dados ocorre aqui: busca-se as propriedades com base na marca selecionada
        self.teor_rotulo_atual = DADOS_CACHACAS[self.marca_atual]["teor_rotulo"]
        self.teor_medido_atual = DADOS_CACHACAS[self.marca_atual]["teor_medido"]
        self.acucar_atual = DADOS_CACHACAS[self.marca_atual]["nivel_acucar"]
        
        self.pureza_atual = self.selecao_pureza.get()
        self.temp_externa_atual = self.entrada_temp_ext.get().strip() 

        # Validações de preenchimento obrigatório
        if not self.temp_externa_atual:
            messagebox.showwarning("Restrição", "O campo de temperatura é obrigatório.")
            return

        try:
            quantidade = int(self.entrada_leituras.get().strip())
            if quantidade <= 0:
                raise ValueError
            self.leituras_restantes = quantidade
            self.leituras_por_serie = quantidade
        except ValueError:
            messagebox.showwarning("Restrição", "A quantidade de leituras deve ser um número inteiro válido e maior que zero.")
            return

        try:
            series = int(self.entrada_series.get().strip())
            if series <= 0:
                raise ValueError
            self.series_restantes = series
        except ValueError:
            messagebox.showwarning("Restrição", "A quantidade de séries deve ser um número inteiro válido e maior que zero.")
            return

        # Bloqueia a interface para evitar interferência do usuário durante a coleta
        self.botao_iniciar.config(state=tk.DISABLED, bg="#aaaaaa", text="⏳ AQUISIÇÃO EM ANDAMENTO...")
        self.registrar_atividade(f"\n--- Solicitando {self.leituras_restantes} Amostra(s): {self.marca_atual} ---")
        self.registrar_atividade(f"--- Séries programadas: {self.series_restantes} ---")
        
        self.enviar_sinal_requisicao()

    def enviar_sinal_requisicao(self):
        """Envia o caractere identificador que instrui a placa a realizar uma varredura capacitiva."""
        if self.conexao_serial and self.conexao_serial.is_open:
            self.conexao_serial.write(b'I')

    def pausar_entre_series(self):
        """Pausa a operação após uma série, transformando o botão principal para aguardar o usuário."""
        # Altera o botão para o modo de continuação
        self.botao_iniciar.config(state=tk.NORMAL, bg="#FF9800", text="▶ CONTINUAR SÉRIE", command=self.continuar_serie)
        self.registrar_atividade("\n[PAUSA] Série concluída. Reposicione a amostra.")
        self.registrar_atividade(f"Você pode ajustar a temperatura. Restam {self.series_restantes} série(s).")

    def continuar_serie(self):
        """Captura a nova temperatura e retoma a próxima série de aquisição."""
        nova_temp = self.entrada_temp_ext.get().strip()
        if not nova_temp:
            messagebox.showwarning("Restrição", "O campo de temperatura é obrigatório.")
            return
            
        self.temp_externa_atual = nova_temp
        
        # Retorna o botão ao estado desabilitado de leitura
        self.botao_iniciar.config(state=tk.DISABLED, bg="#aaaaaa", text="⏳ AQUISIÇÃO EM ANDAMENTO...")
        self.registrar_atividade(f"\n--- Retomando: Série de {self.leituras_por_serie} leitura(s) ---")
        self.enviar_sinal_requisicao()

    def finalizar_ciclo(self):
        """Restaura o estado da interface após a conclusão de todas as leituras programadas."""
        self.entrada_temp_ext.delete(0, tk.END)
        # Retorna o botão ao estado inicial verde
        self.botao_iniciar.config(state=tk.NORMAL, bg="#4CAF50", text="▶ INICIAR COLETA", command=self.iniciar_processo_coleta)
        self.registrar_atividade("--- Ciclo total de coletas e séries finalizado ---")

    def monitorar_porta_continuamente(self):
        """Linha de execução isolada (paralela) que processa as mensagens oriundas do equipamento."""
        while self.execucao_ativa:
            try:
                if self.conexao_serial and self.conexao_serial.in_waiting > 0:
                    linha = self.conexao_serial.readline().decode('utf-8').strip()
                    if not list(linha):
                        continue
                    
                    # Identificador de envio contínuo da temperatura interna do hardware
                    if linha.startswith("RT_TEMP:"):
                        temp_texto = self.entrada_temp_ext.get().strip() + " °C"
                        self.raiz.after(0, lambda t=temp_texto: self.rotulo_status_valor.config(text=t))
                    
                    # Identificador de envio do pacote contendo as capacitâncias medidas
                    elif linha.startswith("DATA:"):
                        dados_brutos = linha.replace("DATA:", "").strip()
                        vetor_dados = dados_brutos.split(",")
                        
                        if len(vetor_dados) > 0:
                            # Sobrescreve o primeiro valor (espaço reservado) com a temperatura do usuário
                            vetor_dados[0] = self.temp_externa_atual
                            
                        dados_estruturados = ";".join(vetor_dados)
                        carimbo_tempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Concatenação da matriz final a ser escrita no disco (inclui propriedades automáticas)
                        linha_csv = f"{carimbo_tempo};{self.marca_atual};{self.teor_rotulo_atual};{self.teor_medido_atual};{self.acucar_atual};{dados_estruturados};{self.pureza_atual}\n"
                        
                        with open(ARQUIVO_CSV, 'a', encoding='utf-8') as arquivo:
                            arquivo.write(linha_csv)
                        
                        self.raiz.after(0, lambda: self.registrar_atividade(f"[SUCESSO] Dado salvo no CSV. (Restam: {self.leituras_restantes - 1})"))
                        
                        # Lógica de decisão para encadeamento de leituras e séries
                        if self.leituras_restantes > 1:
                            self.leituras_restantes -= 1
                            # Sinal sonoro curto e agudo, confirmando a captura intermediária
                            winsound.Beep(1000, 200) 
                            # Agenda a próxima captura com 1 segundo de intervalo para evitar congestionamento
                            self.raiz.after(1000, self.enviar_sinal_requisicao)
                        else:
                            # Fim de uma série. Verifica se há mais séries programadas.
                            if self.series_restantes > 1:
                                self.series_restantes -= 1
                                self.leituras_restantes = self.leituras_por_serie
                                
                                # Três bipes de alerta exigidos
                                winsound.Beep(1500, 300)
                                time.sleep(0.1)
                                winsound.Beep(1500, 300)
                                time.sleep(0.1)
                                winsound.Beep(1500, 300)
                                
                                self.raiz.after(0, self.pausar_entre_series)
                            else:
                                # Sinal sonoro longo, indicando término total do bloco programado
                                winsound.Beep(1500, 800) 
                                self.raiz.after(0, self.finalizar_ciclo)
                    
                    # Repassa quaisquer mensagens de depuração vindas do controlador
                    elif linha:
                        self.raiz.after(0, lambda l=linha: self.registrar_atividade(l))
            except Exception:
                break
            time.sleep(0.05)

if __name__ == "__main__":
    janela_principal = tk.Tk()
    aplicativo = AplicativoColetor(janela_principal)
    
    def encerramento_seguro():
        """Garante a finalização do processo paralelo e liberação da porta física."""
        aplicativo.execucao_ativa = False
        if aplicativo.conexao_serial and aplicativo.conexao_serial.is_open:
            aplicativo.conexao_serial.close()
        janela_principal.destroy()
        
    janela_principal.protocol("WM_DELETE_WINDOW", encerramento_seguro)
    janela_principal.mainloop()
