"""
SISTEMA DE ANÁLISE DE DADOS EM MODELOS DE CLASSIFICAÇÃO
Otimizado para rigor metodológico experimental: Nested Cross-Validation, 
Predições Out-of-Fold, Avaliação Geométrica, com log integral.
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

from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict, train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel 
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
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
st.set_page_config(page_title="Classificação", layout="wide")

# =============================================================================
# 1. MALHA DE HIPERPARÂMETROS (PRESETS)
# =============================================================================
def obter_parametros_malha(nome_modelo, nivel):
    if nivel == "Leve":
        if nome_modelo == "Regressão Logística": return {'motor__C': [1.0], 'motor__solver': ['lbfgs'], 'motor__class_weight': [None]}
        if nome_modelo == "SVM": return {'motor__kernel': ['rbf'], 'motor__C': [1], 'motor__gamma': ['scale'], 'motor__class_weight': [None]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50], 'motor__max_depth': [None], 'motor__min_samples_split': [2], 'motor__class_weight': [None]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [50], 'motor__learning_rate': [0.1], 'motor__max_depth': [3], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(50, 50)], 'motor__activation': ['relu'], 'motor__solver': ['adam'], 'motor__alpha': [0.0001]}
    elif nivel == "Médio":
        if nome_modelo == "Regressão Logística": return {'motor__C': [0.01, 1.0, 100.0], 'motor__solver': ['lbfgs'], 'motor__class_weight': [None, 'balanced']}
        if nome_modelo == "SVM": return {'motor__kernel': ['rbf', 'linear'], 'motor__C': [0.1, 1, 10], 'motor__gamma': ['scale', 'auto'], 'motor__class_weight': [None]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50, 100, 200], 'motor__max_depth': [None, 5, 10], 'motor__min_samples_split': [2], 'motor__class_weight': [None]}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [50, 100, 200], 'motor__learning_rate': [0.01, 0.1], 'motor__max_depth': [3, 5], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'motor__activation': ['relu', 'tanh'], 'motor__solver': ['adam', 'lbfgs'], 'motor__alpha': [0.0001]}
    elif nivel == "Pesado":
        if nome_modelo == "Regressão Logística": return {'motor__C': [0.01, 0.1, 1.0, 10.0, 100.0], 'motor__solver': ['lbfgs'], 'motor__class_weight': [None, 'balanced']}
        if nome_modelo == "SVM": return {'motor__kernel': ['rbf', 'linear', 'poly'], 'motor__C': [0.1, 1, 10, 100], 'motor__gamma': ['scale', 'auto', 0.01, 0.1], 'motor__class_weight': [None]}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [100, 200, 500], 'motor__max_depth': [None, 3, 5, 10, 20], 'motor__min_samples_split': [2], 'motor__class_weight': [None, 'balanced']}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [100, 200, 500], 'motor__learning_rate': [0.01, 0.05, 0.1], 'motor__max_depth': [3, 5, 10], 'motor__subsample': [1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(100,), (50, 50), (100, 50), (100, 100)], 'motor__activation': ['relu', 'tanh'], 'motor__solver': ['adam', 'lbfgs'], 'motor__alpha': [0.0001]}
    elif nivel == "Insano":
        if nome_modelo == "Regressão Logística": return {'motor__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 'motor__solver': ['lbfgs', 'liblinear'], 'motor__class_weight': [None, 'balanced']}
        if nome_modelo == "SVM": return {'motor__kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'motor__C': [0.001, 0.01, 0.1, 1, 10, 100], 'motor__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], 'motor__class_weight': [None, 'balanced']}
        if nome_modelo == "Random Forest": return {'motor__n_estimators': [50, 100, 300, 500, 1000], 'motor__max_depth': [None, 3, 5, 10, 20, 30], 'motor__min_samples_split': [2, 5, 10], 'motor__class_weight': [None, 'balanced', 'balanced_subsample']}
        if nome_modelo == "Gradient Boosting": return {'motor__n_estimators': [100, 300, 500, 1000], 'motor__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], 'motor__max_depth': [3, 5, 10, 20], 'motor__subsample': [0.7, 0.8, 0.9, 1.0]}
        if nome_modelo == "ANN": return {'motor__hidden_layer_sizes': [(50,), (100,), (200,), (50, 50), (100, 100), (50, 50, 50)], 'motor__activation': ['relu', 'tanh', 'logistic'], 'motor__solver': ['adam', 'lbfgs', 'sgd'], 'motor__alpha': [0.0001, 0.001, 0.01, 0.1]}
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
    
    elementos.append(Paragraph("Relatório Técnico de Modelagem e Auditoria (Classificação)", estilos['Title']))
    elementos.append(Spacer(1, 10))
    elementos.append(Paragraph("1. Desempenho Global (Nested CV / Teste Cego)", estilos['Heading2']))
    
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
            fig_box = px.box(df_friedman, points="all", title="Distribuição do F1-Macro nos Folds de Avaliação")
            fig_box.update_layout(yaxis_title="F1-Macro Score", xaxis_title="Modelos Analisados")
            imagem_bytes_friedman = fig_box.to_image(format="png", engine="kaleido")
            elementos.append(ImagemRL(io.BytesIO(imagem_bytes_friedman), width=500, height=300))
            elementos.append(Spacer(1, 15))
        except Exception as e:
            elementos.append(Paragraph(f"[Aviso: Gráfico de Friedman falhou - {e}]", estilo_registro))
        elementos.append(PageBreak())

    for nome, info in info_todos_modelos.items():
        elementos.append(Paragraph(f"Diagnóstico Analítico: {nome}", estilos['Heading2']))
        try:
            if info['matriz_confusao'] is not None:
                fig_cm = px.imshow(info['matriz_confusao'], text_auto=True, color_continuous_scale='Blues', x=info['rotulos_matriz'], y=info['rotulos_matriz'])
                imagem_bytes = fig_cm.to_image(format="png", engine="kaleido")
                elementos.append(ImagemRL(io.BytesIO(imagem_bytes), width=350, height=350))
        except Exception as e:
            elementos.append(Paragraph(f"[Aviso: Gráfico falhou - {e}]", estilo_registro))
        
        if info['tabela_erros'] is not None and not info['tabela_erros'].empty:
            elementos.append(Spacer(1, 15))
            elementos.append(Paragraph(f"Auditoria de Falsos Positivos/Negativos ({len(info['tabela_erros'])} falhas)", estilos['Heading3']))
            tabela_truncada = truncar_tabela(info['tabela_erros'])
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
        
        mapa_siglas = {"Regressão Logística": "Log", "Naive Bayes": "NB", "SVM": "SVM", "Random Forest": "RF", "ANN": "ANN", "Gradient Boosting": "GB"}
        
        for nome_modelo, info in info_todos_modelos.items():
            sigla = mapa_siglas.get(nome_modelo, nome_modelo[:5])
            if info['tabela_erros'] is not None and not info['tabela_erros'].empty:
                info['tabela_erros'].to_excel(escritor, sheet_name=f'Err_{sigla}', index=False)
            if info['tabela_completa'] is not None and not info['tabela_completa'].empty:
                info['tabela_completa'].to_excel(escritor, sheet_name=f'Base_{sigla}', index=False)
    buffer_memoria.seek(0)
    return buffer_memoria

# =============================================================================
# 3. INTERFACE DE USUÁRIO - SIDEBAR
# =============================================================================
st.sidebar.header("0. Base de Dados")
arquivo_carregado = st.sidebar.file_uploader("Arquivo CSV (Treinamento)", type=["csv"])

if arquivo_carregado is None:
    st.title("Sistema Analítico: Classificação Discreta")
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
percentual_separacao = st.sidebar.slider("Geração de Hold-Out Estratificado (%)", 0, 50, 20)

# =============================================================================
# 4. QUALIDADE E METADADOS
# =============================================================================
def avaliar_condicoes_base(tabela_referencia, atritubos_lista, var_alvo, num_folds):
    relatorio = "--- AVALIAÇÃO DE COMPLEXIDADE DOS DADOS ---\n"
    
    freq_classes = tabela_referencia[var_alvo].value_counts(normalize=True) * 100
    contagens = tabela_referencia[var_alvo].value_counts()
    min_class_count = contagens.min()
    
    resumo_freq = " | ".join([f"{k}: {v:.1f}%" for k, v in freq_classes.items()])
    relatorio += f"Espectro Frequencial (Classes): {resumo_freq}\n"
    relatorio += f"↳ Frequência da classe minoritária: {min_class_count} amostras.\n"
    
    if min_class_count < num_folds:
        relatorio += f"⚠️ RISCO ESTÁTISTICO SEVERO: A classe minoritária possui menos amostras ({min_class_count}) do que dobras da validação cruzada ({num_folds}).\n"
    
    if len(atritubos_lista) > 0:
        try:
            matriz_espacial = tabela_referencia[atritubos_lista].values
            condicao_numerica = np.linalg.cond(matriz_espacial)
            vetor_vif = [variance_inflation_factor(matriz_espacial, i) for i in range(matriz_espacial.shape[1])]
            vif_global = np.mean(vetor_vif) if vetor_vif else float('inf')
            
            relatorio += f"Diagnóstico de Multicolinearidade\n  - Condition Number (CN): {condicao_numerica:.2f} - CN > 30 são consideradas colineares, sinalizando instabilidade matricial.\n"
            relatorio += f"  - Mean Variance Inflation Factor (VIF): {vif_global:.2f} - quanto maior o valor (acima de 5), maior a redundância de dados.\n"
            
            if tabela_referencia[var_alvo].nunique() > 1:
                silhueta = silhouette_score(matriz_espacial, tabela_referencia[var_alvo])
                relatorio += f"Separabilidade Geométrica das Classes (Silhouette Score): {silhueta:.3f}\n"
        except Exception as erro_matriz:
            relatorio += f"Falha na decomposição matricial: {erro_matriz}\n"
            
    return relatorio + "\n"

# =============================================================================
# 5. NÚCLEO DE EXECUÇÃO E GUIAS (TABS)
# =============================================================================
st.title("Avaliação de Modelos de Classificação")
guia_config, guia_modelo, guia_diagnostico, guia_auditoria, guia_friedman = st.tabs(["⚙️ Configuração de Execução", "🚀 Execução e Desempenho", "📊 Matriz de Confusão", "🔍 Diagnóstico de Predição", "📈 Teste de Friedman"])

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
        if st.checkbox("Regressão Logística", value=True, key="chk_clf_lr"):
            with st.expander("Parâmetros (Reg. Logística)", expanded=False):
                dp = obter_parametros_malha("Regressão Logística", preset)
                p_C = st.multiselect("C (Inverso Regularização)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], default=dp.get('motor__C', [1.0]), key="clf_lr_C")
                p_solver = st.multiselect("solver", ['lbfgs', 'liblinear', 'saga'], default=dp.get('motor__solver', ['lbfgs']), key="clf_lr_sol")
                p_cw = st.multiselect("class_weight", [None, 'balanced'], default=dp.get('motor__class_weight', [None]), key="clf_lr_cw")
                
                if len(p_C)>0 and len(p_solver)>0 and len(p_cw)>0:
                    fila_modelos.append(("Regressão Logística", LogisticRegression(max_iter=1000, multi_class='auto'), {'motor__C': p_C, 'motor__solver': p_solver, 'motor__class_weight': p_cw}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")
        
        if st.checkbox("SVM", value=True, key="chk_clf_svm"):
            with st.expander("Parâmetros (SVM)", expanded=False):
                dp = obter_parametros_malha("SVM", preset)
                p_ker = st.multiselect("kernel", ['rbf', 'linear', 'poly', 'sigmoid'], default=dp.get('motor__kernel', ['rbf']), key="clf_svm_ker")
                p_C = st.multiselect("C", [0.001, 0.01, 0.1, 1, 10, 100], default=dp.get('motor__C', [1]), key="clf_svm_C")
                p_gam = st.multiselect("gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], default=dp.get('motor__gamma', ['scale']), key="clf_svm_gam")
                p_cw = st.multiselect("class_weight", [None, 'balanced'], default=dp.get('motor__class_weight', [None]), key="clf_svm_cw")
                
                if len(p_ker)>0 and len(p_C)>0 and len(p_gam)>0 and len(p_cw)>0:
                    fila_modelos.append(("SVM", SVC(probability=True, random_state=42), {'motor__kernel': p_ker, 'motor__C': p_C, 'motor__gamma': p_gam, 'motor__class_weight': p_cw}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

    with c2:
        if st.checkbox("Random Forest", value=True, key="chk_clf_rf"):
            with st.expander("Parâmetros (Random Forest)", expanded=False):
                dp = obter_parametros_malha("Random Forest", preset)
                p_est = st.multiselect("n_estimators", [50, 100, 200, 300, 500, 1000], default=dp.get('motor__n_estimators', [50]), key="clf_rf_est")
                p_md = st.multiselect("max_depth", [None, 3, 5, 10, 20, 30], default=dp.get('motor__max_depth', [None]), key="clf_rf_md")
                p_mss = st.multiselect("min_samples_split", [2, 5, 10], default=dp.get('motor__min_samples_split', [2]), key="clf_rf_mss")
                p_cw = st.multiselect("class_weight", [None, 'balanced', 'balanced_subsample'], default=dp.get('motor__class_weight', [None]), key="clf_rf_cw")
                
                if len(p_est)>0 and len(p_md)>0 and len(p_mss)>0 and len(p_cw)>0:
                    fila_modelos.append(("Random Forest", RandomForestClassifier(random_state=42), {'motor__n_estimators': p_est, 'motor__max_depth': p_md, 'motor__min_samples_split': p_mss, 'motor__class_weight': p_cw}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")
        
        if st.checkbox("Gradient Boosting", value=True, key="chk_clf_gb"):
            with st.expander("Parâmetros (Gradient Boosting)", expanded=False):
                dp = obter_parametros_malha("Gradient Boosting", preset)
                p_est = st.multiselect("n_estimators", [50, 100, 200, 300, 500, 1000], default=dp.get('motor__n_estimators', [50]), key="clf_gb_est")
                p_lr = st.multiselect("learning_rate", [0.001, 0.01, 0.05, 0.1, 0.2], default=dp.get('motor__learning_rate', [0.1]), key="clf_gb_lr")
                p_md = st.multiselect("max_depth", [3, 5, 10, 20], default=dp.get('motor__max_depth', [3]), key="clf_gb_md")
                p_sub = st.multiselect("subsample", [0.7, 0.8, 0.9, 1.0], default=dp.get('motor__subsample', [1.0]), key="clf_gb_sub")
                
                if len(p_est)>0 and len(p_lr)>0 and len(p_md)>0 and len(p_sub)>0:
                    fila_modelos.append(("Gradient Boosting", GradientBoostingClassifier(random_state=42), {'motor__n_estimators': p_est, 'motor__learning_rate': p_lr, 'motor__max_depth': p_md, 'motor__subsample': p_sub}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")

    with c3:
        if st.checkbox("ANN (Rede Neural)", value=True, key="chk_clf_ann"):
            with st.expander("Parâmetros (ANN)", expanded=False):
                dp = obter_parametros_malha("ANN", preset)
                p_hls = st.multiselect("hidden_layer_sizes", [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100), (50, 50, 50)], default=dp.get('motor__hidden_layer_sizes', [(50, 50)]), key="clf_ann_hls")
                p_act = st.multiselect("activation", ['relu', 'tanh', 'logistic'], default=dp.get('motor__activation', ['relu']), key="clf_ann_act")
                p_sol = st.multiselect("solver", ['adam', 'lbfgs', 'sgd'], default=dp.get('motor__solver', ['adam']), key="clf_ann_sol")
                p_alp = st.multiselect("alpha", [0.0001, 0.001, 0.01, 0.1], default=dp.get('motor__alpha', [0.0001]), key="clf_ann_alp")
                
                if len(p_hls)>0 and len(p_act)>0 and len(p_sol)>0 and len(p_alp)>0:
                    fila_modelos.append(("ANN", MLPClassifier(max_iter=2000, random_state=42, early_stopping=True), {'motor__hidden_layer_sizes': p_hls, 'motor__activation': p_act, 'motor__solver': p_sol, 'motor__alpha': p_alp}, None))
                else: st.warning("Selecione ao menos um valor em cada parâmetro.")
        
        if st.checkbox("Naive Bayes", value=True, key="chk_clf_nb"):
            st.info("Naive Bayes não possui hiperparâmetros ajustáveis nesta malha. Será rodado isoladamente.")
            fila_modelos.append(("Naive Bayes", GaussianNB(), {}, "Padrão"))

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

        # 1. Hold-Out Interno Estratificado
        fracao_teste = percentual_separacao / 100.0
        if fracao_teste > 0:
            df_tv, df_teste_int = train_test_split(df_tv, test_size=fracao_teste, random_state=42, stratify=df_tv[coluna_alvo])
            lista_df_teste.append(df_teste_int.copy())
            texto_holdout_partes.append(f"Split Interno ({percentual_separacao}%)")

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
            ref_dados_cego = df_teste_consolidado 
        else:
            atrib_teste = alvo_teste = None
            texto_holdout = "Nenhuma validação cega"
            ref_dados_cego = None

        cv_externa = StratifiedKFold(n_splits=qtd_dobras, shuffle=True, random_state=42)
        cv_interna = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        aviso_interface = st.empty() 
        tabela_resultados = [] 
        
        texto_filtros = " | ".join([f"{k}: {v}" for k, v in regras_filtros.items()]) if regras_filtros else "Livre"
        texto_extracao = f"{metodo_extracao} (Reter={limite_atributos})" if metodo_extracao != "Nenhuma" else "Desabilitada"
        
        registro_textual = f"--- DADOS DA EXECUÇÃO ---\nData/Hora: {st.session_state['data_hora_execucao'].strftime('%d/%m/%Y %H:%M')}\n"
        registro_textual += f"Target: {coluna_alvo}\nFeatures: {', '.join(atributos_operacionais)}\n"
        registro_textual += f"Dados de aprendizagem: {len(atrib_tv)}  | Dados externos de teste: {len(atrib_teste) if atrib_teste is not None else 0}\n"
        registro_textual += f"Cross-Validation: {qtd_dobras} Folds\n\n"
        registro_textual += avaliar_condicoes_base(base_processamento, atributos_operacionais, coluna_alvo, qtd_dobras)
        registro_textual += "--- AVALIAÇÃO DE MODELOS ---\n"
        registro_textual += "MÉTRICAS DE DESEMPENHO:\n - F1-Macro: média harmônica entre precisão e recall (calculada para cada classe e promediada); excelente para medir eficiência em bases desbalanceadas.\n - Acc (Accuracy): proporção de amostras classificadas corretamente; mais intuitiva, porém sensível a classes majoritárias.\n - B-Acc (Balanced Accuracy): média do recall de cada classe individualmente; ajusta a percepção de acertos compensando distorções de frequência de classe.\n\nDIAGNÓSTICO ESPACIAL E DE CLASSES:\n - Matriz de Confusão: tabulação cruzada das predições corretas e incorretas detalhadas pela realidade amostral.\n - Frequência e Silhouette Score: análise da dominância das classes e da sobreposição vetorial geométrica das predições.\n\n"

        texto_diarios = ""

        def executar_modelo(rotulo, objeto_modelo, dict_parametros=None, ref_manual=None):
            raio_x_iteracao = f"--- {rotulo} ---\n"
            etapas_fluxo = [("padronizador", StandardScaler())]
            
            if metodo_extracao != "Nenhuma" and limite_atributos < len(atributos_operacionais):
                seletor = SelectKBest(mutual_info_classif, k=limite_atributos) if "Informação" in metodo_extracao else SelectFromModel(RandomForestClassifier(random_state=42), max_features=limite_atributos, threshold=-np.inf)
                etapas_fluxo.append(("filtro_dimensao", seletor))
                
            etapas_fluxo.append(("motor", objeto_modelo))
            fluxo_base = Pipeline(etapas_fluxo)
            
            # Análise de Modo: Isolado vs Grid
            usa_grid = any(len(v) > 1 for v in dict_parametros.values()) if dict_parametros else False
            
            t_zero = time.time()

            if usa_grid:
                aviso_interface.warning(f"Otimizando {rotulo} via GridSearch (Nested CV)...")
                otimizador_grid = GridSearchCV(fluxo_base, dict_parametros, cv=cv_interna, scoring='f1_macro', n_jobs=-1, verbose=3)
                validacao = cross_validate(otimizador_grid, atrib_tv, alvo_tv, cv=cv_externa, scoring={'acc': 'accuracy', 'f1': 'f1_macro', 'bacc': 'balanced_accuracy'}, return_estimator=True, n_jobs=4, verbose=3)
                otimizador_grid.fit(atrib_tv, alvo_tv)
                motor_campeao = otimizador_grid.best_estimator_
                conf_campea = str({k.replace('motor__', ''): v for k, v in otimizador_grid.best_params_.items()})
                
                res_grid = otimizador_grid.cv_results_
                for mean_score, params in zip(res_grid['mean_test_score'], res_grid['params']):
                    p_clean = {k.replace('motor__', ''): v for k, v in params.items()}
                    raio_x_iteracao += f" * Tentativa {p_clean} -> F1-Macro = {mean_score:.4f}\n"
            else:
                aviso_interface.info(f"Avaliando {rotulo} em execução única (CV Externo)...")
                if dict_parametros:
                    single_params_flow = {k: v[0] for k, v in dict_parametros.items()}
                    fluxo_base.set_params(**single_params_flow)
                    conf_campea = str({k.replace('motor__', ''): v for k, v in single_params_flow.items()})
                else:
                    conf_campea = ref_manual if ref_manual else "Padrão Absoluto"
                    
                validacao = cross_validate(fluxo_base, atrib_tv, alvo_tv, cv=cv_externa, scoring={'acc': 'accuracy', 'f1': 'f1_macro', 'bacc': 'balanced_accuracy'}, return_estimator=True)
                motor_campeao = fluxo_base.fit(atrib_tv, alvo_tv)
                raio_x_iteracao += f" * Execução Única: {conf_campea} -> Concluída.\n"
                
            tempo_decorrido = time.time() - t_zero
            acc_cv, f1_cv, bacc_cv = np.mean(validacao['test_acc']), np.mean(validacao['test_f1']), np.mean(validacao['test_bacc'])
            
            # --- TESTE DE FRIEDMAN ---
            if repeticoes_friedman > 0:
                validador_est = StratifiedShuffleSplit(n_splits=repeticoes_friedman, test_size=0.2, random_state=42)
                scores_f1_folds = cross_val_score(motor_campeao, atrib_tv, alvo_tv, cv=validador_est, scoring='f1_macro', n_jobs=-1)
            else:
                scores_f1_folds = validacao['test_f1']
            
            if metodo_extracao != "Nenhuma":
                freq_features = {f: 0 for f in atributos_operacionais}
                for estimador in validacao['estimator']:
                    modelo_treinado = estimador.best_estimator_ if hasattr(estimador, 'best_estimator_') else estimador
                    suporte = modelo_treinado.named_steps['filtro_dimensao'].get_support()
                    for idx, mantida in enumerate(suporte):
                        if mantida: freq_features[atributos_operacionais[idx]] += 1
                
                raio_x_iteracao += "\nEstabilidade das Features (Frequência de Seleção nos Folds):\n"
                for feat, contagem in freq_features.items():
                    if contagem > 0: raio_x_iteracao += f"  - {feat}: {(contagem/qtd_dobras)*100:.1f}%\n"

            str_matriz = ""
            if atrib_teste is not None:
                alvo_predito = motor_campeao.predict(atrib_teste) 
                y_proba = motor_campeao.predict_proba(atrib_teste) if hasattr(motor_campeao, "predict_proba") else None
                classes_modelo = motor_campeao.classes_ if hasattr(motor_campeao, "classes_") else None

                acc_cego, f1_cego, bacc_cego = accuracy_score(alvo_teste, alvo_predito), f1_score(alvo_teste, alvo_predito, average='macro'), balanced_accuracy_score(alvo_teste, alvo_predito)
                cm, rotulos_matriz = confusion_matrix(alvo_teste, alvo_predito), np.unique(np.concatenate((alvo_teste, alvo_predito)))
                
                base_completa = atrib_teste.copy()
                base_completa['Real (Gabarito)'] = alvo_teste
                base_completa['Classe Prevista'] = alvo_predito
                
                if y_proba is not None:
                    base_completa['Probabilidade Estimada (%)'] = (np.max(y_proba, axis=1) * 100).round(2)
                    for idx, cls_name in enumerate(classes_modelo):
                        base_completa[f'Probab: {cls_name} (%)'] = (y_proba[:, idx] * 100).round(2)
                else:
                    base_completa['Probabilidade Estimada (%)'] = "N/A"

                for c in colunas_categoricas: 
                    base_completa[c] = ref_dados_cego[c].values
                base_completa.reset_index(drop=True, inplace=True); base_completa.index += 1
                base_erros = base_completa[base_completa['Real (Gabarito)'] != base_completa['Classe Prevista']]
                
                str_matriz = f"\n  Distribuição de Acertos (Matriz de Confusão do Teste Cego):\n    Eixos (Classes): {rotulos_matriz.tolist()}\n"
                for i_linha, linha in enumerate(cm): str_matriz += f"    Real {rotulos_matriz[i_linha]}: {linha.tolist()}\n"
            else:
                alvo_predito_tv = cross_val_predict(motor_campeao, atrib_tv, alvo_tv, cv=cv_externa)
                cm, rotulos_matriz = confusion_matrix(alvo_tv, alvo_predito_tv), np.unique(np.concatenate((alvo_tv, alvo_predito_tv)))
                acc_cego, f1_cego, bacc_cego = np.nan, np.nan, np.nan
                base_completa, base_erros = None, None
                
                str_matriz = f"\n  Distribuição de Acertos (Matriz Out-of-Fold / Validação Cruzada):\n    Eixos (Classes): {rotulos_matriz.tolist()}\n"
                for i_linha, linha in enumerate(cm): str_matriz += f"    Real {rotulos_matriz[i_linha]}: {linha.tolist()}\n"

            info_white_box, modelo_interno = "", motor_campeao.named_steps['motor']
            if rotulo == "Regressão Logística":
                try:
                    pesos = modelo_interno.coef_[0]
                    feats = [atributos_operacionais[i] for i, v in enumerate(motor_campeao.named_steps['filtro_dimensao'].get_support()) if v] if 'filtro_dimensao' in motor_campeao.named_steps else atributos_operacionais.copy()
                    info_white_box = f"  ↳ White-Box (Logística - Pesos): Z = {' + '.join([f'({p:.3f} * {f})' for p, f in zip(pesos, feats)])}\n"
                except: pass

            sumario = f"> Modelo: {rotulo}\n"
            sumario += f"  Melhores Hiperparâmetros: {conf_campea}\n"
            sumario += f"  Tempo de execução: {tempo_decorrido:.2f}s\n"
            sumario += f"  Validação Cruzada: F1-Macro = {f1_cv:.4f} | Acc = {acc_cv:.4f} | B-Acc = {bacc_cv:.4f}\n"
            if atrib_teste is not None: 
                sumario += f"  Validação Cega: F1-Macro = {f1_cego:.4f} | Acc = {acc_cego:.4f} | B-Acc = {bacc_cego:.4f}\n"
            sumario += info_white_box + str_matriz

            f1_h, acc_h, bacc_h = np.nan, np.nan, np.nan
            f1_ext, acc_ext, bacc_ext = np.nan, np.nan, np.nan

            if df_teste_int is not None:
                pred_h = motor_campeao.predict(df_teste_int[atributos_operacionais])
                alvo_h = df_teste_int[coluna_alvo]
                acc_h, f1_h, bacc_h = accuracy_score(alvo_h, pred_h), f1_score(alvo_h, pred_h, average='macro'), balanced_accuracy_score(alvo_h, pred_h)

            if base_externa_proc is not None:
                pred_ext = motor_campeao.predict(base_externa_proc[atributos_operacionais])
                alvo_ext = base_externa_proc[coluna_alvo]
                acc_ext, f1_ext, bacc_ext = accuracy_score(alvo_ext, pred_ext), f1_score(alvo_ext, pred_ext, average='macro'), balanced_accuracy_score(alvo_ext, pred_ext)

            f1_campeao = f1_ext if base_externa_proc is not None else (f1_h if df_teste_int is not None else f1_cv)
            pacote_modelo = {'nome': rotulo, 'f1_rank': f1_campeao, 'tabela_erros': base_erros, 'tabela_completa': base_completa, 'matriz_confusao': cm, 'rotulos_matriz': rotulos_matriz, 'scores_cv': scores_f1_folds}

            dict_resultado = {
                "Modelo Analisado": rotulo, "Acc (CV)": acc_cv, "BAcc (CV)": bacc_cv, "F1-Macro (CV)": f1_cv
            }
            if df_teste_int is not None:
                dict_resultado.update({"Acc (Holdout)": acc_h, "BAcc (Holdout)": bacc_h, "F1 (Holdout)": f1_h})
            if base_externa_proc is not None:
                dict_resultado.update({"Acc (Externo)": acc_ext, "BAcc (Externo)": bacc_ext, "F1 (Externo)": f1_ext})
            
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
            matriz_f1 = {n: dict_info_modelos[n]['scores_cv'] for n in nomes_modelos}
            df_friedman = pd.DataFrame(matriz_f1)
            
            lista_para_teste = [matriz_f1[n] for n in nomes_modelos]
            qui2_est, valor_p = friedmanchisquare(*lista_para_teste)
            
            # Ajuste dinâmico do texto com base na metodologia executada
            metodologia_blocos = f"Stratified Shuffle Split ({repeticoes_friedman} rodadas)" if repeticoes_friedman > 0 else f"K-Fold CV Externa ({qtd_dobras} Folds)"
            
            log_estatistico += f"Metodologia (Blocos): {metodologia_blocos}\n"
            log_estatistico += f"Grupos Avaliados: {', '.join(nomes_modelos)}\n"
            log_estatistico += f"Estatística de Teste: {qui2_est:.3f} | Valor-p: {valor_p:.4e}\n"
            log_estatistico += "Hipótese Nula (H0): Os algoritmos possuem distribuições de desempenho (F1-Macro) estatisticamente equivalentes.\n"
            
            if valor_p < 0.05:
                log_estatistico += "Decisão: Rejeita-se H0. Há evidências estatísticas de superioridade de desempenho de um ou mais modelos.\n"
            else:
                log_estatistico += "Decisão: Falha em rejeitar H0. Não há evidências estatísticas suficientes para declarar superioridade de um modelo sobre os demais.\n"
            log_estatistico += "\n"

        m_g, s_g = divmod(time.time() - relogio_inicial, 60)
        registro_textual += texto_sumarios + log_estatistico + f"\nTempo Computacional: {int(m_g)}m {s_g:.1f}s\n" + "-"*40 + "\n\nRAIO-X COMBINATÓRIO\n" + texto_diarios
        
        coluna_sort = "F1 (Externo)" if usar_base_externa else ("F1 (Holdout)" if percentual_separacao > 0 else "F1-Macro (CV)")
        base_ordenada = pd.DataFrame(tabela_resultados).sort_values(by=coluna_sort, ascending=False).reset_index(drop=True)
        
        st.session_state.update({'base_ordenada': base_ordenada, 'registro_textual': registro_textual, 'dict_info_modelos': dict_info_modelos, 'df_friedman': df_friedman, 'tem_teste': atrib_teste is not None, 'finalizado': True})
        aviso_interface.success("Ciclo Finalizado com Rigor Experimental.")

if st.session_state.get('finalizado', False):
    st.sidebar.markdown("---")
    st.sidebar.header("5. Extração de Resultados")
    
    nome_padrao = st.session_state['data_hora_execucao'].strftime('%Y%m%d-%H%M')
    if TEM_EXCEL:
        arq_excel = gerar_excel_geral(st.session_state['base_ordenada'], st.session_state['dict_info_modelos'])
        st.sidebar.download_button("📥 Extrair Auditoria Completa (XLSX)", data=arq_excel, file_name=f"{nome_padrao}-Classificacao.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if TEM_PDF:
        arq_pdf = gerar_relatorio_pdf_amplo(st.session_state['registro_textual'], st.session_state['base_ordenada'], st.session_state['dict_info_modelos'],df_friedman=st.session_state.get('df_friedman') )
        st.sidebar.download_button("📥 Extrair Relatório Metodológico (PDF)", data=arq_pdf, file_name=f"{nome_padrao}-Classificacao.pdf", mime="application/pdf")

    with guia_modelo:
        st.dataframe(st.session_state['base_ordenada'].style.format(precision=4), width='stretch')
        st.text_area("Diário de Execução (Log Técnico)", value=st.session_state['registro_textual'], height=600)

    with guia_diagnostico:
        mod_diag = st.selectbox("Modelo a Diagnosticar:", list(st.session_state['dict_info_modelos'].keys()))
        info_cm = st.session_state['dict_info_modelos'][mod_diag]
        if info_cm['matriz_confusao'] is not None:
            titulo_grafico = "Matriz de Confusão (Teste Cego)" if st.session_state['tem_teste'] else "Matriz de Confusão (Out-of-Fold CV)"
            grafico_cm = px.imshow(info_cm['matriz_confusao'], text_auto=True, color_continuous_scale='Blues', x=info_cm['rotulos_matriz'], y=info_cm['rotulos_matriz'], labels=dict(x="Classe Prevista", y="Real (Gabarito)", color="Frequência"), title=titulo_grafico)
            st.plotly_chart(grafico_cm, width='stretch')

    with guia_auditoria:
        mod_auditoria = st.selectbox("Inspecionar Falhas do Modelo:", list(st.session_state['dict_info_modelos'].keys()), key="auditoria_select")
        base_erros = st.session_state['dict_info_modelos'][mod_auditoria]['tabela_erros']
        
        if base_erros is not None and not base_erros.empty:
            st.dataframe(base_erros, width='stretch')
        elif base_erros is not None:
            st.success("Operação Estável. O modelo não apresentou divergências de previsão na malha testada.")
        else:
            st.info("A auditoria linha a linha está disponível apenas ao utilizar a Validação Cega (separação de teste).")

    with guia_friedman:
        df_f = st.session_state.get('df_friedman')
        if df_f is not None and not df_f.empty:
            fig_box = px.box(df_f, points="all", title="Estabilidade do F1-Macro nos Folds de Avaliação")
            st.plotly_chart(fig_box, width='stretch')
