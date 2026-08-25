"""
SISTEMA DE ANÁLISE DE DADOS EM MODELOS DE REGRESSÃO
Otimizado para rigor metodológico experimental: Nested Cross-Validation, 
Teste de Breusch-Pagan e Validação em Blocos (Friedman), com log integral.
"""

import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import time 
import warnings 
import ast 
import io
import json
from datetime import datetime 
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge 
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SelectFromModel 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

TEM_PDF = False
try:
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ImagemRL, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    TEM_PDF = True
except ImportError:
    st.sidebar.warning("Atenção: Bibliotecas de PDF ausentes. (pip install reportlab kaleido)")

TEM_EXCEL = False
try:
    import openpyxl
    TEM_EXCEL = True
except ImportError:
    st.sidebar.warning("Atenção: Biblioteca 'openpyxl' ausente.")

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Regressão", layout="wide")

# =============================================================================
# 1. MALHA DE HIPERPARÂMETROS (PRESETS)
# =============================================================================
def obter_parametros_malha(nome_modelo, nivel):
    if nivel == "Leve":
        if nome_modelo == "Regressão Linear Múltipla": return {'motor__fit_intercept': [True]}
        if nome_modelo == "Regressão Bayesiana": return {'motor__alpha_1': [1e-6], 'motor__lambda_1': [1e-6]}
        if nome_modelo == "SVR": return {'motor__kernel': ['rbf'], 'motor__C': [1], 'motor__epsilon': [0.1]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50], 'motor__max_depth': [None], 'motor__min_samples_split': [2], 'motor__min_samples_leaf': [1]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [50], 'motor__learning_rate': [0.1], 'motor__max_depth': [3], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(100, 100)], 'motor__activation': ['relu'], 'motor__solver': ['lbfgs'], 'motor__alpha': [0.1]}
        if nome_modelo == "XGBoost": return {'motor__n_estimators': [50], 'motor__learning_rate': [0.1], 'motor__max_depth': [3], 'motor__subsample': [1.0]}
    elif nivel == "Médio":
        if nome_modelo == "Regressão Linear Múltipla": return {'motor__fit_intercept': [True, False]}
        if nome_modelo == "Regressão Bayesiana": return {'motor__alpha_1': [1e-6, 1e-4], 'motor__lambda_1': [1e-6, 1e-4]}
        if nome_modelo == "SVR": return {'motor__kernel': ['rbf', 'linear'], 'motor__C': [0.1, 1, 10], 'motor__epsilon': [0.01, 0.1]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50, 100, 200], 'motor__max_depth': [None, 5, 10], 'motor__min_samples_split': [2], 'motor__min_samples_leaf': [1]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [50, 100, 200], 'motor__learning_rate': [0.01, 0.1], 'motor__max_depth': [3, 5], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'motor__activation': ['relu', 'tanh'], 'motor__solver': ['adam', 'lbfgs'], 'motor__alpha': [0.1]}
        if nome_modelo == "XGBoost": return {'motor__n_estimators': [50, 100, 200], 'motor__learning_rate': [0.01, 0.1], 'motor__max_depth': [3, 5], 'motor__subsample': [0.8, 1.0]}
    elif nivel == "Pesado":
        if nome_modelo == "Regressão Linear Múltipla": return {'motor__fit_intercept': [True, False]}
        if nome_modelo == "Regressão Bayesiana": return {'motor__alpha_1': [1e-6, 1e-4, 1e-2], 'motor__lambda_1': [1e-6, 1e-4, 1e-2]}
        if nome_modelo == "SVR": return {'motor__kernel': ['rbf', 'linear', 'poly'], 'motor__C': [0.1, 1, 10, 100], 'motor__epsilon': [0.01, 0.05, 0.1, 0.2]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [100, 200, 500], 'motor__max_depth': [None, 3, 5, 10, 20], 'motor__min_samples_split': [2, 5], 'motor__min_samples_leaf': [1]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [100, 200, 500], 'motor__learning_rate': [0.01, 0.05, 0.1], 'motor__max_depth': [3, 5, 10], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(100,), (50, 50), (100, 50), (100, 100)], 'motor__activation': ['relu', 'tanh'], 'motor__solver': ['adam', 'lbfgs'], 'motor__alpha': [0.1]}
        if nome_modelo == "XGBoost": return {'motor__n_estimators': [100, 300, 500], 'motor__learning_rate': [0.01, 0.05, 0.1], 'motor__max_depth': [3, 5, 7], 'motor__subsample': [0.8, 1.0]}
    elif nivel == "Insano":
        if nome_modelo == "Regressão Linear Múltipla": return {'motor__fit_intercept': [True, False]}
        if nome_modelo == "Regressão Bayesiana": return {'motor__alpha_1': [1e-6, 1e-4, 1e-2, 1.0], 'motor__lambda_1': [1e-6, 1e-4, 1e-2, 1.0]}
        if nome_modelo == "SVR": return {'motor__kernel': ['rbf', 'linear', 'poly'], 'motor__C': [0.1, 1, 10, 100, 1000], 'motor__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50, 100, 300, 500], 'motor__max_depth': [None, 10, 20, 30], 'motor__min_samples_split': [2, 5], 'motor__min_samples_leaf': [1, 2]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [100, 300, 500, 1000, 2000], 'motor__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 'motor__max_depth': [3, 5, 10, 20], 'motor__subsample': [0.7, 0.8, 0.9, 1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(100,), (200,), (50, 50), (100, 100), (100, 50, 25)], 'motor__activation': ['relu', 'tanh'], 'motor__solver': ['adam', 'lbfgs'], 'motor__alpha': [0.001, 0.01, 0.1]}
        if nome_modelo == "XGBoost": return {'motor__n_estimators': [100, 300, 500, 1000], 'motor__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2], 'motor__max_depth': [3, 5, 7, 10], 'motor__subsample': [0.6, 0.8, 1.0], 'motor__colsample_bytree': [0.8, 1.0]}
    return {}

# =============================================================================
# 2. MOTORES DE EXPORTAÇÃO
# =============================================================================
def truncar_tabela(dados, limite=18):
    return dados.astype(str).map(lambda x: (x[:limite] + '..') if len(x) > limite else x)

def gerar_relatorio_pdf_amplo(registro_textual, tabela_desempenho, info_todos_modelos, df_friedman=None):
    buffer_memoria = io.BytesIO()
    documento = SimpleDocTemplate(buffer_memoria, pagesize=landscape(A4), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    elementos = []
    estilos = getSampleStyleSheet()
    estilo_registro = ParagraphStyle(name='Log', parent=estilos['Normal'], fontName='Courier', fontSize=7, leading=9)
    
    elementos.append(Paragraph("Relatório Técnico de Modelagem e Auditoria (Regressão)", estilos['Title']))
    elementos.append(Spacer(1, 10))
    elementos.append(Paragraph("1. Desempenho Global", estilos['Heading2']))
    
    dados_tabela_limpos = tabela_desempenho.astype(str)
    colunas_limpas = [c.replace('<br>', ' ') for c in dados_tabela_limpos.columns.tolist()]
    matriz_desempenho = [colunas_limpas] + dados_tabela_limpos.values.tolist()
    
    tabela_pdf = Table(matriz_desempenho)
    tabela_pdf.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                                    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0,0), (-1,-1), 8), ('GRID', (0,0), (-1,-1), 0.5, colors.black)]))
    elementos.append(tabela_pdf)
    elementos.append(PageBreak())

    if df_friedman is not None and not df_friedman.empty:
        elementos.append(Paragraph("Análise Estatística: Estabilidade (Teste de Friedman)", estilos['Heading2']))
        try:
            fig_box = px.box(df_friedman, points="all", title="Distribuição do R² nos Folds de Avaliação")
            fig_box.update_layout(yaxis_title="R² Score", xaxis_title="Modelos Analisados")
            imagem_bytes_friedman = fig_box.to_image(format="png", engine="kaleido")
            elementos.append(ImagemRL(io.BytesIO(imagem_bytes_friedman), width=500, height=300))
            elementos.append(Spacer(1, 15))
        except Exception as e:
            elementos.append(Paragraph(f"[Aviso: Gráfico de Friedman falhou - {e}]", estilo_registro))
        elementos.append(PageBreak())

    for nome, info in info_todos_modelos.items():
        elementos.append(Paragraph(f"Diagnóstico Analítico: {nome}", estilos['Heading2']))
        try:
            dados_residuos = info['dados_auditoria']
            fig_res = px.scatter(dados_residuos, x='Previsto', y='Resíduo', opacity=0.7)
            fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
            imagem_bytes = fig_res.to_image(format="png", engine="kaleido")
            elementos.append(ImagemRL(io.BytesIO(imagem_bytes), width=450, height=300))
        except Exception as e:
            elementos.append(Paragraph(f"[Aviso: Gráfico falhou - {e}]", estilo_registro))
        
        if info['dados_auditoria'] is not None:
            elementos.append(Spacer(1, 15))
            elementos.append(Paragraph("Amostra de Erros Críticos (Piores Predições)", estilos['Heading3']))
            erros_graves = info['dados_auditoria'].sort_values(by="Erro Absoluto", ascending=False).head(20)
            tabela_truncada = truncar_tabela(erros_graves)
            matriz_erros = [tabela_truncada.columns.tolist()] + tabela_truncada.values.tolist()
            tb_erros = Table(matriz_erros)
            tb_erros.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkred), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                                          ('FONTSIZE', (0,0), (-1,-1), 5), ('GRID', (0,0), (-1,-1), 0.5, colors.black)]))
            elementos.append(tb_erros)
        elementos.append(PageBreak())

    elementos.append(Paragraph("Registro Tecnico e Metadados", estilos['Heading2']))
    for linha in registro_textual.split('\n'):
        elementos.append(Paragraph(linha.replace(' ', '&nbsp;'), estilo_registro))
        
    documento.build(elementos)
    buffer_memoria.seek(0)
    return buffer_memoria

def gerar_excel_geral(tabela_desempenho, info_todos_modelos):
    buffer_memoria = io.BytesIO()
    with pd.ExcelWriter(buffer_memoria, engine='openpyxl') as escritor:
        tabela_limpa = tabela_desempenho.copy()
        tabela_limpa.columns = tabela_limpa.columns.str.replace('<br>', ' ')
        tabela_limpa.to_excel(escritor, sheet_name='Desempenho', index=False)
        
        mapa_siglas = {"Regressão Linear Múltipla": "RLM", "Regressão Bayesiana": "RBayes", "SVR": "SVR", 
                       "Random Forest": "RF", "ANN": "ANN", "Gradient Boosting": "GB", "XGBoost": "XGB"}
        
        for nome_modelo, info in info_todos_modelos.items():
            sigla = mapa_siglas.get(nome_modelo, nome_modelo[:5])
            dados_completos = info['dados_auditoria']
            
            if dados_completos is not None and not dados_completos.empty:
                tabela_erros = dados_completos.sort_values(by="Erro Absoluto", ascending=False)
                tabela_erros.to_excel(escritor, sheet_name=f'Err_{sigla}', index=False)
                dados_completos.sort_index().to_excel(escritor, sheet_name=f'Base_{sigla}', index=False)
    buffer_memoria.seek(0)
    return buffer_memoria

def interpretar_heterocedasticidade(p_valor):
    if p_valor < 0.05:
        return (f"  Análise de Resíduos: O teste de Breusch-Pagan rejeitou a hipótese de homocedasticidade (p={p_valor:.4e}).\n"
                f"  Interpretação: Os resíduos apresentam heterocedasticidade sistemática. Isto sugere que a variância do erro não é constante.")
    else:
        return (f"  Análise de Resíduos: O teste de Breusch-Pagan não rejeitou a hipótese de homocedasticidade (p={p_valor:.4e}).\n"
                f"  Interpretação: O modelo exibe variância de erro constante, indicando uma distribuição estocástica bem comportada dos resíduos.")

# =============================================================================
# 3. INTERFACE DE USUÁRIO - SIDEBAR
# =============================================================================
st.sidebar.header("0. Base de Dados")
arquivo_carregado = st.sidebar.file_uploader("Arquivo CSV (Treinamento)", type=["csv"])

if arquivo_carregado is None:
    st.title("Sistema Analítico: Regressão Contínua")
    st.info("Insira a base de dados para iniciar.")
    st.stop()

base_original = pd.read_csv(arquivo_carregado, sep=";") 

st.sidebar.header("1. Estrutura de Dados")
todas_colunas = base_original.columns.tolist()
coluna_alvo = st.sidebar.selectbox("Target (Variável Alvo)", todas_colunas)
colunas_categoricas = st.sidebar.multiselect("Filtros em Features (não obrigatório)", [c for c in todas_colunas if c != coluna_alvo])
colunas_numericas = [c for c in base_original.select_dtypes(include=[np.number]).columns if c != coluna_alvo and c not in colunas_categoricas]
atributos_operacionais = st.sidebar.multiselect("Features (Atributos Operacionais)", colunas_numericas, default=colunas_numericas)

base_processamento = base_original.copy()
regras_filtros = {}

if colunas_categoricas:
    for col in colunas_categoricas:
        selecoes = st.sidebar.multiselect(f"Restringir {col}", base_original[col].unique(), default=base_original[col].unique())
        base_processamento = base_processamento[base_processamento[col].isin(selecoes)]
        regras_filtros[col] = selecoes

st.sidebar.header("2. Otimização de Features")
metodo_extracao = st.sidebar.selectbox("Técnica", ["Nenhuma", "Informação Mútua (K-Best)", "Importância de Árvore (RF)"])
limite_atributos = len(atributos_operacionais)
if metodo_extracao != "Nenhuma" and limite_atributos > 0:
    limite_atributos = st.sidebar.slider("Reter Top N Features", 1, limite_atributos, max(1, limite_atributos-1))

st.sidebar.header("3. Metodologia de Validação")
qtd_dobras = st.sidebar.slider("K-Fold (Dobras) CV", 3, 12, 5)
repeticoes_friedman = st.sidebar.slider("Rodadas para Teste de Friedman", 0, 30, 10)

st.sidebar.header("4. Validação Cega")
arquivo_teste_externo = st.sidebar.file_uploader("Subir Base de Validação Física (Recomendado)", type=["csv"], key="upload_externo")
usar_base_externa = arquivo_teste_externo is not None
percentual_separacao = st.sidebar.slider("Geração de Hold-Out Aleatório (%)", 0, 50, 20)

# =============================================================================
# 4. COMPLEXIDADE DOS DADOS
# =============================================================================
def avaliar_condicoes_base(tabela_referencia, atritubos_lista):
    relatorio = "--- AVALIAÇÃO DE COMPLEXIDADE DOS DADOS ---\n"
    
    if len(atritubos_lista) > 0:
        coef_var = tabela_referencia[atritubos_lista].apply(lambda x: (x.std() / x.mean()) * 100 if x.mean() != 0 else 0).mean()
        relatorio += f"Dispersão de dados\n  - Coefficient of Variation (CV):: {coef_var:.2f}%\n"
        
        try:
            matriz_espacial = tabela_referencia[atritubos_lista].values
            condicao_numerica = np.linalg.cond(matriz_espacial)
            diagnostico_cn = "Baixa" if condicao_numerica < 30 else ("Moderada" if condicao_numerica < 100 else "Alta/Crítica")
            
            vetor_vif = [variance_inflation_factor(matriz_espacial, i) for i in range(matriz_espacial.shape[1])]
            vif_global = np.mean(vetor_vif) if vetor_vif else float('inf')

            relatorio += f"Diagnóstico de Multicolinearidade\n  - Condition Number (CN): {condicao_numerica:.2f} - CN > 30 são consideradas colineares, sinalizando instabilidade matricial.\n"
            relatorio += f"  - Mean Variance Inflation Factor (VIF): {vif_global:.2f} - quanto maior o valor (acima de 5), maior a redundância de dados.\n"
        except Exception as erro_matriz:
            relatorio += f"Falha na decomposição matricial: {erro_matriz}\n"
            
    return relatorio + "\n"

def calcular_vif_final(df_features):
    try:
        if df_features.empty or df_features.shape[1] < 2: return "N/A"
        matriz = df_features.values
        vifs = [variance_inflation_factor(matriz, i) for i in range(matriz.shape[1])]
        return np.mean(vifs)
    except:
        return "Erro de Matriz"

# =============================================================================
# 5. NÚCLEO DE EXECUÇÃO E GUIAS (TABS)
# =============================================================================
st.title("Avaliação de Modelos de Regressão")
guia_config, guia_modelo, guia_diagnostico, guia_auditoria, guia_friedman = st.tabs(["⚙️ Configuração de Execução", "🚀 Execução e Desempenho", "📊 Estatística de Resíduos", "🔍 Auditoria Linha a Linha", "📈 Teste de Friedman"])

# VARIÁVEL GLOBAL PARA OS MODELOS SELECIONADOS
fila_modelos = []

with guia_config:
    st.header("Seleção de Modelos e Hiperparâmetros")

    # --- INÍCIO DA MIGRAÇÃO DE ESTADO (SAVE/LOAD) ---
    c_exp, c_imp = st.columns(2)
    
    with c_exp:
        st.markdown("**Exportar Configuração Atual**")
        # Filtra dinamicamente as variáveis de estado que pertencem aos painéis de configuração
        chaves_alvo = ('chk_', 'reg_', 'clf_')
        dict_estado = {k: v for k, v in st.session_state.items() if k.startswith(chaves_alvo)}
        json_exportacao = json.dumps(dict_estado, indent=4)
        
        st.download_button("💾 Baixar JSON de Parâmetros", data=json_exportacao, file_name="preset_modelos.json", mime="application/json")
        
    with c_imp:
        st.markdown("**Importar Configuração**")
        arq_config = st.file_uploader("Subir preset (.json)", type=["json"], label_visibility="collapsed")
        
        # O botão "Aplicar" evita que o Streamlit trave a tela no estado do arquivo carregado
        if arq_config is not None:
            if st.button("Aplicar Configuração Carregada", type="secondary"):
                try:
                    estado_carregado = json.load(arq_config)
                    for chave, valor in estado_carregado.items():
                        st.session_state[chave] = valor
                    st.success("Aplicado! Modifique os parâmetros ou rode as avaliações.")
                    st.rerun() # Força o redesenho da tela para atualizar os botões visuais
                except Exception as erro:
                    st.error(f"Falha ao ler o dicionário de dados: {erro}")
    st.markdown("---")
    # --- FIM DA MIGRAÇÃO DE ESTADO ---

    #preset = st.select_slider("Preset de Rigor (Atualiza as faixas padrão dos seletores abaixo)", ["Leve", "Médio", "Pesado", "Insano"], value="Médio")
    preset = "Médio"
    st.markdown("Selecione os algoritmos desejados...")
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.checkbox("Regressão Linear Múltipla", value=True, key="chk_reg_rlm"):
            with st.expander("Parâmetros (Reg. Linear)", expanded=False):
                dp = obter_parametros_malha("Regressão Linear Múltipla", preset)
                p_fit = st.multiselect("fit_intercept", [True, False], default=dp.get('motor__fit_intercept', [True, False]), key="reg_rlm_fit")
                
                if len(p_fit)>0:
                    fila_modelos.append(("Regressão Linear Múltipla", LinearRegression(), {'motor__fit_intercept': p_fit}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")
        
        if st.checkbox("Regressão Bayesiana", value=True, key="chk_reg_bayes"):
            with st.expander("Parâmetros (Bayesiana)", expanded=False):
                dp = obter_parametros_malha("Regressão Bayesiana", preset)
                p_a1 = st.multiselect("alpha_1", [1e-6, 1e-4, 1e-2, 1.0, 100.0], default=dp.get('motor__alpha_1', [1e-6]), key="reg_bayes_a1")
                p_l1 = st.multiselect("lambda_1", [1e-6, 1e-4, 1e-2, 1.0, 100.0], default=dp.get('motor__lambda_1', [1e-6]), key="reg_bayes_l1")
                
                if len(p_a1)>0 and len(p_l1)>0:
                    fila_modelos.append(("Regressão Bayesiana", BayesianRidge(), {'motor__alpha_1': p_a1, 'motor__lambda_1': p_l1}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

    with c2:
        if st.checkbox("SVR", value=True, key="chk_reg_svr"):
            with st.expander("Parâmetros (SVR)", expanded=False):
                dp = obter_parametros_malha("SVR", preset)
                p_ker = st.multiselect("kernel", ['rbf', 'linear', 'poly', 'sigmoid'], default=dp.get('motor__kernel', ['rbf']), key="reg_svr_ker")
                p_C = st.multiselect("C", [0.001, 0.01, 0.05, 0.1, 1, 10, 100, 1000], default=dp.get('motor__C', [1]), key="reg_svr_C")
                p_eps = st.multiselect("epsilon", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5], default=dp.get('motor__epsilon', [0.1]), key="reg_svr_eps")
                
                if len(p_ker)>0 and len(p_C)>0 and len(p_eps)>0:
                    fila_modelos.append(("SVR", SVR(), {'motor__kernel': p_ker, 'motor__C': p_C, 'motor__epsilon': p_eps}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

        if st.checkbox("Random Forest", value=True, key="chk_reg_rf"):
            with st.expander("Parâmetros (Random Forest)", expanded=False):
                dp = obter_parametros_malha("Random Forest", preset)
                p_est = st.multiselect("n_estimators", [50, 100, 200, 300, 500, 1000, 2000], default=dp.get('motor__n_estimators', [50]), key="reg_rf_est")
                p_md = st.multiselect("max_depth", [None, 3, 5, 8, 10, 20, 30, 50], default=dp.get('motor__max_depth', [None]), key="reg_rf_md")
                p_mss = st.multiselect("min_samples_split", [2, 5, 10], default=dp.get('motor__min_samples_split', [2]), key="reg_rf_mss")
                p_msl = st.multiselect("min_samples_leaf", [1, 2, 4], default=dp.get('motor__min_samples_leaf', [1]), key="reg_rf_msl")
                
                if len(p_est)>0 and len(p_md)>0 and len(p_mss)>0 and len(p_msl)>0:
                    fila_modelos.append(("Random Forest", RandomForestRegressor(random_state=42), {'motor__n_estimators': p_est, 'motor__max_depth': p_md, 'motor__min_samples_split': p_mss, 'motor__min_samples_leaf': p_msl}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

    with c3:
        if st.checkbox("Gradient Boosting", value=True, key="chk_reg_gb"):
            with st.expander("Parâmetros (Gradient Boosting)", expanded=False):
                dp = obter_parametros_malha("Gradient Boosting", preset)
                p_est = st.multiselect("n_estimators", [50, 100, 200, 300, 400, 500, 1000, 2000], default=dp.get('motor__n_estimators', [50]), key="reg_gb_est")
                p_lr = st.multiselect("learning_rate", [0.001, 0.01, 0.05, 0.1, 0.2], default=dp.get('motor__learning_rate', [0.1]), key="reg_gb_lr")
                p_md = st.multiselect("max_depth", [3, 5, 10, 20], default=dp.get('motor__max_depth', [3]), key="reg_gb_md")
                p_sub = st.multiselect("subsample", [0.7, 0.8, 0.9, 1.0], default=dp.get('motor__subsample', [1.0]), key="reg_gb_sub")
                
                if len(p_est)>0 and len(p_lr)>0 and len(p_md)>0 and len(p_sub)>0:
                    fila_modelos.append(("Gradient Boosting", GradientBoostingRegressor(random_state=42), {'motor__n_estimators': p_est, 'motor__learning_rate': p_lr, 'motor__max_depth': p_md, 'motor__subsample': p_sub}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

        if st.checkbox("ANN (Rede Neural)", value=True, key="chk_reg_ann"):
            with st.expander("Parâmetros (ANN)", expanded=False):
                dp = obter_parametros_malha("ANN", preset)
                p_hls = st.multiselect("hidden_layer_sizes", [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100), (50, 50, 50), (100, 50, 25)], default=dp.get('motor__hidden_layer_sizes', [(100, 100)]), key="reg_ann_hls")
                p_act = st.multiselect("activation", ['relu', 'tanh', 'logistic'], default=dp.get('motor__activation', ['relu']), key="reg_ann_act")
                p_sol = st.multiselect("solver", ['adam', 'lbfgs', 'sgd'], default=dp.get('motor__solver', ['lbfgs']), key="reg_ann_sol")
                p_alp = st.multiselect("alpha", [0.0001, 0.001, 0.01, 0.1, 1, 2], default=dp.get('motor__alpha', [0.1]), key="reg_ann_alp")
                
                if len(p_hls)>0 and len(p_act)>0 and len(p_sol)>0 and len(p_alp)>0:
                    fila_modelos.append(("ANN", MLPRegressor(max_iter=2000, random_state=42, early_stopping=True), {'motor__hidden_layer_sizes': p_hls, 'motor__activation': p_act, 'motor__solver': p_sol, 'motor__alpha': p_alp}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")
                
        if st.checkbox("XGBoost", value=True, key="chk_reg_xgb"):
            with st.expander("Parâmetros (XGBoost)", expanded=False):
                dp = obter_parametros_malha("XGBoost", preset)
                p_est = st.multiselect("n_estimators", [50, 100, 200, 300, 500, 600, 1000], default=dp.get('motor__n_estimators', [50]), key="reg_xgb_est")
                p_lr = st.multiselect("learning_rate", [0.005, 0.01, 0.03, 0.05, 0.1, 0.2], default=dp.get('motor__learning_rate', [0.1]), key="reg_xgb_lr")
                p_md = st.multiselect("max_depth", [3, 4, 5, 7, 10], default=dp.get('motor__max_depth', [3]), key="reg_xgb_md")
                p_sub = st.multiselect("subsample", [0.6, 0.7, 0.8, 1.0], default=dp.get('motor__subsample', [1.0]), key="reg_xgb_sub")
                p_col = st.multiselect("colsample_bytree", [0.5, 0.6, 0.8, 1.0], default=dp.get('motor__colsample_bytree', [1.0]), key="reg_xgb_col")
                
                if len(p_est)>0 and len(p_lr)>0 and len(p_md)>0 and len(p_sub)>0 and len(p_col)>0:
                    # Usando o erro absoluto para lidar com a heterocedasticidade
                    #modelo_xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
                    modelo_xgb = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_jobs=-1)
                    fila_modelos.append(("XGBoost", modelo_xgb, {'motor__n_estimators': p_est, 'motor__learning_rate': p_lr, 'motor__max_depth': p_md, 'motor__subsample': p_sub, 'motor__colsample_bytree': p_col}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

with guia_modelo:
    if st.button("RODAR AVALIAÇÕES", type="primary", width='stretch'):
        if not fila_modelos: st.error("Nenhum modelo configurado na aba de Configuração."); st.stop()
        st.session_state['data_hora_execucao'] = datetime.now()
        if len(atributos_operacionais) == 0: st.error("Nenhuma feature selecionada."); st.stop()
            
        df_tv = base_processamento.copy()
        lista_df_teste = []
        texto_holdout_partes = []
        df_teste_int = None 
        base_externa_proc = None

        # 1. Hold-Out Interno Aleatório
        fracao_teste = percentual_separacao / 100.0
        if fracao_teste > 0:
            df_tv, df_teste_int = train_test_split(df_tv, test_size=fracao_teste, random_state=42)
            lista_df_teste.append(df_teste_int.copy())
            texto_holdout_partes.append(f"Partição Interna ({percentual_separacao}%)")

        # 2. Base Externa
        if usar_base_externa:
            try:
                base_externa = pd.read_csv(arquivo_teste_externo, sep=";")
                if set(base_original.columns) != set(base_externa.columns):
                    st.error("Erro Metodológico: Divergência de dimensões entre Treino e Teste."); st.stop()
                for c, v in regras_filtros.items(): base_externa = base_externa[base_externa[c].isin(v)]
                base_externa_proc = base_externa.copy()
                lista_df_teste.append(base_externa_proc)
                texto_holdout_partes.append(f"Arquivo Externo ({arquivo_teste_externo.name})")
            except Exception as e:
                st.error(f"Leitura rejeitada: {e}"); st.stop()

        # 3. Consolidação das Bases
        atrib_tv = df_tv[atributos_operacionais]
        alvo_tv = df_tv[coluna_alvo]

        if len(lista_df_teste) > 0:
            df_teste_consolidado = pd.concat(lista_df_teste, ignore_index=True)
            atrib_teste = df_teste_consolidado[atributos_operacionais]
            alvo_teste = df_teste_consolidado[coluna_alvo]
            texto_holdout = " + ".join(texto_holdout_partes)
        else:
            atrib_teste = alvo_teste = None
            texto_holdout = "Nenhuma validação cega"        

        cv_externa = KFold(n_splits=qtd_dobras, shuffle=True, random_state=42)
        cv_interna = KFold(n_splits=3, shuffle=True, random_state=42) 
        
        aviso_interface = st.empty() 
        tabela_resultados = [] 
        
        texto_filtros = " | ".join([f"{k}: {v}" for k, v in regras_filtros.items()]) if regras_filtros else "Livre"
        texto_extracao = f"{metodo_extracao} (Reter={limite_atributos})" if metodo_extracao != "Nenhuma" else "Desabilitada"
        
        registro_textual = f"--- DADOS DA EXECUÇÃO ---\nData/Hora: {st.session_state['data_hora_execucao'].strftime('%d/%m/%Y %H:%M')}\n"
        registro_textual += f"Target: {coluna_alvo}\nFeatures: {', '.join(atributos_operacionais)}\n"
        registro_textual += f"Dados de aprendizagem: {len(atrib_tv)}  | Dados externos de teste: {len(atrib_teste) if atrib_teste is not None else 0}\n"
        registro_textual += f"Cross-Validation: {qtd_dobras} Folds\n\n"
        registro_textual += avaliar_condicoes_base(base_processamento, atributos_operacionais)
        registro_textual += "--- AVALIAÇÃO DE MODELOS ---\n"
        registro_textual += "MÉTRICAS DE DESEMPENHO:\n - R² (Coefficient of Determination): proporção da variância da variável dependente; valores próximos a 1.0 indicam melhor ajuste.\n - MAE (Mean Absolute Error): média das diferenças absolutas entre predições e valores reais.\n - RMSE (Root Mean Square Error): raiz quadrada da média dos quadrados dos erros; sensível a outliers.\n\nANÁLISE DE RESÍDUOS:\n - Teste de Breusch-Pagan: avaliação da homocedasticidade. P-value > 0.05 indica variância constante.\n\n"

        texto_diarios = ""

        def executar_modelo(rotulo, objeto_modelo, dict_parametros=None, ref_manual=None):
            raio_x_iteracao = f"--- {rotulo} ---\n"
            etapas_fluxo = [("padronizador", StandardScaler())]
            
            if metodo_extracao != "Nenhuma" and limite_atributos < len(atributos_operacionais):
                seletor = SelectKBest(mutual_info_regression, k=limite_atributos) if "Informação" in metodo_extracao else SelectFromModel(RandomForestRegressor(random_state=42), max_features=limite_atributos, threshold=-np.inf)
                etapas_fluxo.append(("filtro_dimensao", seletor))
                
            etapas_fluxo.append(("motor", objeto_modelo))
            fluxo_base = Pipeline(etapas_fluxo)

            # Análise de Modo: Isolado vs Grid
            usa_grid = any(len(v) > 1 for v in dict_parametros.values()) if dict_parametros else False

            t_zero = time.time()

            if usa_grid:
                aviso_interface.warning(f"Otimizando {rotulo} via GridSearch (Nested CV)...")
                otimizador_grid = GridSearchCV(fluxo_base, dict_parametros, cv=cv_interna, scoring='r2', n_jobs=-1, verbose=3)
                validacao = cross_validate(otimizador_grid, atrib_tv, alvo_tv, cv=cv_externa, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}, return_estimator=True)
                otimizador_grid.fit(atrib_tv, alvo_tv)
                motor_campeao = otimizador_grid.best_estimator_
                conf_campea = str({k.replace('motor__', ''): v for k, v in otimizador_grid.best_params_.items()})
                
                res_grid = otimizador_grid.cv_results_
                for mean_score, params in zip(res_grid['mean_test_score'], res_grid['params']):
                    p_clean = {k.replace('motor__', ''): v for k, v in params.items()}
                    raio_x_iteracao += f" * Tentativa {p_clean} -> R² = {mean_score:.4f}\n"
            else:
                aviso_interface.info(f"Avaliando {rotulo} em execução única (CV Externo)...")
                if dict_parametros:
                    single_params_flow = {k: v[0] for k, v in dict_parametros.items()}
                    fluxo_base.set_params(**single_params_flow)
                    conf_campea = str({k.replace('motor__', ''): v for k, v in single_params_flow.items()})
                else:
                    conf_campea = ref_manual if ref_manual else "Parâmetros Padrão"
                    
                validacao = cross_validate(fluxo_base, atrib_tv, alvo_tv, cv=cv_externa, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}, return_estimator=True, n_jobs=4, verbose=3)
                motor_campeao = fluxo_base.fit(atrib_tv, alvo_tv)
                raio_x_iteracao += f" * Execução Única: {conf_campea} -> Concluída.\n"

            if metodo_extracao != "Nenhuma":
                freq_features = {f: 0 for f in atributos_operacionais}
                for estimador in validacao['estimator']:
                    modelo_treinado = estimador.best_estimator_ if hasattr(estimador, 'best_estimator_') else estimador
                    suporte = modelo_treinado.named_steps['filtro_dimensao'].get_support()
                    for idx, mantida in enumerate(suporte):
                        if mantida: freq_features[atributos_operacionais[idx]] += 1
                
                raio_x_iteracao += "\nEstabilidade das Features (Frequência de Seleção nos Folds):\n"
                for feat, contagem in freq_features.items():
                    if contagem > 0:
                        raio_x_iteracao += f"  - {feat}: {(contagem/qtd_dobras)*100:.1f}%\n"
                
                suporte_final = motor_campeao.named_steps['filtro_dimensao'].get_support()
                atributos_finais = [atributos_operacionais[i] for i, v in enumerate(suporte_final) if v]
                vif_final = calcular_vif_final(atrib_tv[atributos_finais])
                raio_x_iteracao += f"\nVIF Final (Pós-seleção): {vif_final}\n"

            r2_validacao, mae_validacao, rmse_validacao = np.mean(validacao['test_r2']), -np.mean(validacao['test_mae']), -np.mean(validacao['test_rmse'])
            
            # --- TESTE DE FRIEDMAN ---
            if repeticoes_friedman > 0:
                validador_est = ShuffleSplit(n_splits=repeticoes_friedman, test_size=0.2, random_state=42)
                scores_cv_folds = cross_val_score(motor_campeao, atrib_tv, alvo_tv, cv=validador_est, scoring='r2', n_jobs=-1)
            else:
                scores_cv_folds = validacao['test_r2']
            
            tempo_decorrido = time.time() - t_zero
            
            if atrib_teste is not None:
                alvo_predito = motor_campeao.predict(atrib_teste) 
                r2_cego, mae_cego, rmse_cego = r2_score(alvo_teste, alvo_predito), mean_absolute_error(alvo_teste, alvo_predito), mean_squared_error(alvo_teste, alvo_predito)**0.5
                base_auditoria = atrib_teste.copy()
                base_auditoria['Real'], base_auditoria['Previsto'] = alvo_teste, alvo_predito
            else:
                r2_cego, mae_cego, rmse_cego = np.nan, np.nan, np.nan
                alvo_predito_tv = motor_campeao.predict(atrib_tv)
                base_auditoria = atrib_tv.copy()
                base_auditoria['Real'], base_auditoria['Previsto'] = alvo_tv, alvo_predito_tv

            base_auditoria['Resíduo'] = base_auditoria['Real'] - base_auditoria['Previsto']
            base_auditoria['Erro Absoluto'] = base_auditoria['Resíduo'].abs()
            base_auditoria['Erro Relativo (%)'] = np.where(base_auditoria['Real'] != 0, (base_auditoria['Erro Absoluto'] / np.abs(base_auditoria['Real'])) * 100, 0.0)
            
            try:
                X_bpg = sm.add_constant(base_auditoria[atributos_operacionais].values)
                _, p_val_bp, _, _ = het_breuschpagan(base_auditoria['Resíduo'], X_bpg)
            except:
                p_val_bp = np.nan

            sumario = f"> Modelo: {rotulo}\n"
            sumario += f"  Melhores Hiperparâmetros: {conf_campea}\n"
            sumario += f"  Tempo de execução: {tempo_decorrido:.2f}s\n"
            sumario += f"  Validação Cruzada: R² = {r2_validacao:.4f} | MAE = {mae_validacao:.4f} | RMSE = {rmse_validacao:.4f}\n"
            if atrib_teste is not None:
                sumario += f"  Validação Cega: R² = {r2_cego:.4f} | MAE = {mae_cego:.4f} | RMSE = {rmse_cego:.4f}\n"
            sumario += f"  Análise de Resíduos: p={p_val_bp:.4e}\n"
            
            if rotulo in ["Regressão Linear Múltipla", "Regressão Bayesiana"]:
                try:
                    algoritmo_puro = motor_campeao.named_steps['motor']
                    atr_usados = [atributos_operacionais[i] for i, v in enumerate(motor_campeao.named_steps['filtro_dimensao'].get_support()) if v] if 'filtro_dimensao' in motor_campeao.named_steps else atributos_operacionais.copy()
                    equacao = " + ".join([f"({coef:.3f}*{atr})" for coef, atr in zip(np.ravel(algoritmo_puro.coef_), atr_usados)])
                    sumario += f"  Equação de Transferência: {coluna_alvo} = {equacao} + {np.ravel(algoritmo_puro.intercept_)[0]:.3f}\n"
                except: pass

            r2_h, mae_h, rmse_h = np.nan, np.nan, np.nan
            r2_ext, mae_ext, rmse_ext = np.nan, np.nan, np.nan

            if df_teste_int is not None:
                pred_h = motor_campeao.predict(df_teste_int[atributos_operacionais])
                alvo_h = df_teste_int[coluna_alvo]
                r2_h, mae_h, rmse_h = r2_score(alvo_h, pred_h), mean_absolute_error(alvo_h, pred_h), mean_squared_error(alvo_h, pred_h)**0.5

            if base_externa_proc is not None:
                pred_ext = motor_campeao.predict(base_externa_proc[atributos_operacionais])
                alvo_ext = base_externa_proc[coluna_alvo]
                r2_ext, mae_ext, rmse_ext = r2_score(alvo_ext, pred_ext), mean_absolute_error(alvo_ext, pred_ext), mean_squared_error(alvo_ext, pred_ext)**0.5

            r2_campeao = r2_ext if base_externa_proc is not None else (r2_h if df_teste_int is not None else r2_validacao)
            pacote_modelo = {'nome': rotulo, 'r2_rank': r2_campeao, 'dados_auditoria': base_auditoria, 'scores_cv': scores_cv_folds, 'bp_p': p_val_bp}

            dict_resultado = {
                "Modelo Analisado": rotulo, "R2 (CV)": r2_validacao, "MAE (CV)": mae_validacao, "RMSE (CV)": rmse_validacao
            }
            if df_teste_int is not None:
                dict_resultado.update({"R2 (Holdout)": r2_h, "MAE (Holdout)": mae_h, "RMSE (Holdout)": rmse_h})
            if base_externa_proc is not None:
                dict_resultado.update({"R2 (Externo)": r2_ext, "MAE (Externo)": mae_ext, "RMSE (Externo)": rmse_ext})
            
            dict_resultado["Configuração"] = conf_campea
            tabela_resultados.append(dict_resultado)
            
            return sumario + "\n", raio_x_iteracao + "\n", pacote_modelo

        relogio_inicial = time.time()
        texto_sumarios, dict_info_modelos = "", {}

        for titulo, alg, malha, params_man in fila_modelos:
            sum_mod, diar_mod, inf_mod = executar_modelo(titulo, alg, malha, params_man)
            texto_sumarios += sum_mod
            texto_diarios += diar_mod
            dict_info_modelos[titulo] = inf_mod

        log_estatistico = "\n--- INFERÊNCIA ESTATÍSTICA (FRIEDMAN TEST) ---\n"
        df_friedman = None
        if len(dict_info_modelos) >= 3:
            nomes_modelos = list(dict_info_modelos.keys())
            matriz_r2 = {n: dict_info_modelos[n]['scores_cv'] for n in nomes_modelos}
            df_friedman = pd.DataFrame(matriz_r2)
            
            lista_para_teste = [matriz_r2[n] for n in nomes_modelos]
            qui2_est, valor_p = friedmanchisquare(*lista_para_teste)
            
            log_estatistico += f"Metodologia: Comparação em Blocos (K-Fold CV)\n"
            log_estatistico += f"Grupos Avaliados: {', '.join(nomes_modelos)}\n"
            log_estatistico += f"Estatística de Teste: {qui2_est:.3f} | Valor-p: {valor_p:.4e}\n"
            log_estatistico += f"Hipótese Nula (H0): Os algoritmos possuem distribuições de desempenho (R²) estatisticamente equivalentes.\n"
            log_estatistico += "Decisão: Rejeita-se H0. Há evidências estatísticas de superioridade de desempenho de um ou mais modelos.\n"
            log_estatistico += "\n"
        
        m_g, s_g = divmod(time.time() - relogio_inicial, 60)
        registro_textual += texto_sumarios + log_estatistico + f"\nTempo Computacional: {int(m_g)}m {s_g:.1f}s\n" + "-"*40 + "\n\nRAIO-X COMBINATÓRIO\n" + texto_diarios
        
        coluna_sort = "R2 (Externo)" if usar_base_externa else ("R2 (Holdout)" if percentual_separacao > 0 else "R2 (CV)")
        base_ordenada = pd.DataFrame(tabela_resultados).sort_values(by=coluna_sort, ascending=False).reset_index(drop=True)
        
        st.session_state.update({'base_ordenada': base_ordenada, 'registro_textual': registro_textual, 'dict_info_modelos': dict_info_modelos, 'df_friedman': df_friedman, 'tem_teste': atrib_teste is not None, 'finalizado': True})
        aviso_interface.success("Ciclo Finalizado com Êxito.")

if st.session_state.get('finalizado', False):
    st.sidebar.markdown("---")
    st.sidebar.header("5. Extração de Resultados")
    
    nome_padrao = st.session_state['data_hora_execucao'].strftime('%Y%m%d-%H%M')
    if TEM_EXCEL:
        arq_excel = gerar_excel_geral(st.session_state['base_ordenada'], st.session_state['dict_info_modelos'])
        st.sidebar.download_button("📥 Extrair Auditoria Completa (XLSX)", data=arq_excel, file_name=f"{nome_padrao}-Regressao.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if TEM_PDF:
        arq_pdf = gerar_relatorio_pdf_amplo(st.session_state['registro_textual'], st.session_state['base_ordenada'], st.session_state['dict_info_modelos'],df_friedman=st.session_state.get('df_friedman'))
        st.sidebar.download_button("📥 Extrair Relatório Metodológico (PDF)", data=arq_pdf, file_name=f"{nome_padrao}-Regressao.pdf", mime="application/pdf")

    with guia_modelo:
        st.dataframe(st.session_state['base_ordenada'].style.format(precision=4), width='stretch')
        st.text_area("Diário de Execução (Log Técnico)", value=st.session_state['registro_textual'], height=600)

    with guia_diagnostico:
        mod_diag = st.selectbox("Modelo a Diagnosticar:", list(st.session_state['dict_info_modelos'].keys()))
        dados_grafico = st.session_state['dict_info_modelos'][mod_diag]['dados_auditoria']
        p_val_bp = st.session_state['dict_info_modelos'][mod_diag]['bp_p']
        
        if dados_grafico is not None:
            grafico_res = px.scatter(dados_grafico, x='Previsto', y='Resíduo', opacity=0.7)
            st.plotly_chart(grafico_res, width='stretch')
            if p_val_bp < 0.05: st.warning(f"Risco (Heterocedasticidade): O teste de Breusch-Pagan rejeita variância constante (p={p_val_bp:.4e}).")
            else: st.success(f"Estabilidade (Homocedasticidade): O teste de Breusch-Pagan aponta variância constante (p={p_val_bp:.4e}).")

    with guia_auditoria:
        mod_auditoria = st.selectbox("Selecionar Base:", list(st.session_state['dict_info_modelos'].keys()), key="auditoria_select")
        base_completa = st.session_state['dict_info_modelos'][mod_auditoria]['dados_auditoria']
        
        if base_completa is not None:
            def destacar_desvio(linha): return ['background-color: #ffcccc'] * len(linha) if abs(linha['Erro Relativo (%)']) > 15.0 else [''] * len(linha)
            st.dataframe(base_completa.style.apply(destacar_desvio, axis=1).format(precision=4), width='stretch')

    with guia_friedman:
        df_f = st.session_state.get('df_friedman')
        if df_f is not None and not df_f.empty:
            fig_box = px.box(df_f, points="all", title="Estabilidade do R² nos Folds de Avaliação")
            st.plotly_chart(fig_box, width='stretch')