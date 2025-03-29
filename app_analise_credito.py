import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt


# Configuração de tela 
def config__tela():
    st.set_page_config(
        page_title='Classificação de analise de crédito', 
        page_icon='🏷️',
        layout='wide'
    )
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Classificação Análise de Crédito 🏷️</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #FFD700;'>Análise realizada em duas bases de dados: Treinamento & Teste 🎲</h4>", unsafe_allow_html=True)
    st.markdown("""
    **Desenvolvido por Leandro Souza**  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leandro-souza-bi/)
""")
config__tela()


# Ocultar menus
def ocultar_menu():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
ocultar_menu()


st.divider()


# Ocultar menus
def ocultar_menu():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
ocultar_menu()


@st.cache_data
def carregamento_treino():
    data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/clientes.xlsx"
    df_treino = pd.read_excel(data_url, sheet_name=0)
    df_treino = df_treino.dropna()
    return df_treino
df_treino = carregamento_treino()


@st.cache_data
def carregamento_teste():
    data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/clientes.xlsx"
    df_teste = pd.read_excel(data_url, sheet_name=1)
    return df_teste
df_teste = carregamento_teste()


def codificador():
    codificador_profissao = LabelEncoder()
    df_treino["profissao"] = codificador_profissao.fit_transform(df_treino["profissao"])
    codificador_credito = LabelEncoder()
    df_treino["mix_credito"] = codificador_credito.fit_transform(df_treino["mix_credito"])
    codificador_pagamento = LabelEncoder()
    df_treino["comportamento_pagamento"] = codificador_pagamento.fit_transform(df_treino["comportamento_pagamento"])
    return df_treino, codificador_profissao, codificador_credito, codificador_pagamento

df_treino, codificador_profissao, codificador_credito, codificador_pagamento = codificador()


def conj_treinamento():
    y = df_treino['score_credito']
    x = df_treino.drop(columns=['score_credito','salario_anual'])
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
    return x_treino, x_teste, y_treino, y_teste

x_treino, x_teste, y_treino, y_teste = conj_treinamento()


def modelo_ia():
    modelo_selecionado = RandomForestClassifier()
    modelo_selecionado.fit(x_treino, y_treino)
    return modelo_selecionado

modelo_selecionado = modelo_ia()

def sidebar():
    st.sidebar.markdown("<h2 style='color: #A67C52;'>Sobre o projeto</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    Este projeto utiliza **Machine Learning** para classificar clientes com base em seu **Score de Crédito**.  
    O objetivo é prever o perfil de crédito (Bom, Ruim ou Padrão) a partir de dados financeiros e comportamentais dos clientes.

    ---
    #### 🎯 Modelo Utilizado
    **RandomForestClassifier**  
    Esse modelo funciona como uma "floresta" de decisões. Ele constrói **várias árvores de decisão** durante o treinamento e, na hora de classificar, ele consulta todas essas árvores e define o resultado baseado na **maioria dos votos**.

    É um modelo eficiente, rápido e costuma ter boa performance mesmo em bases complexas, pois reduz o risco de erros de uma única árvore (o famoso **overfitting**).

    ---
    🔍 **Como funciona na prática**
    - Analisa variáveis como: profissão, número de contas, comportamento de pagamento, entre outros.
    - Classifica os clientes em diferentes categorias de Score de Crédito.
    - Ajuda instituições financeiras a tomarem decisões com base no perfil de risco.

    ---
    #### 📊 Validação Visual do Modelo
    Além da acurácia numérica, este projeto também apresenta **métricas visuais** para validar o desempenho do modelo:

    - **Matriz de Confusão:** Mostra, de forma clara, onde o modelo acertou e errou ao classificar os clientes.
    - **Curva ROC (Multiclasse):** Avalia a capacidade do modelo distinguir corretamente entre as categorias de Score de Crédito, exibindo a relação entre a taxa de verdadeiros e falsos positivos.

    Essas análises visuais reforçam a confiabilidade do modelo e ajudam na interpretação dos resultados.
    """)
sidebar()
    


def acuracidade():
    previsao = modelo_selecionado.predict(x_teste)
    acuracia = accuracy_score(y_teste, previsao)
    return acuracia
acuracia = acuracidade()

def card():
    nome_modelo = type(modelo_selecionado).__name__ 
    st.metric(
        label=f'Acuracidade: **{nome_modelo}**',
        value=f"{acuracia:.2%}")
card()


col1, col2 = st.columns(2)

with col1:
    def table_treino():
        st.markdown('## Treinamento')
        table = carregamento_treino()
        table = st.dataframe(table, use_container_width=True, hide_index=True)
        return table
    table = table_treino()
    
with col2:
    def table_teste():
        st.markdown('## Teste')
        table_t = carregamento_teste()
        table_t = st.dataframe(table_t, use_container_width=True, hide_index=True)
        return table_t
    table_t = table_teste()
    
st.divider()

def importancia(modelo_selecionado, x_teste):
    st.markdown("<h4 style='text-align: center; color: #FFD700;'>Colunas por grau de importância %</h4>", unsafe_allow_html=True)
    colunas = list(x_teste.columns)
    importancia = pd.DataFrame(
        index=colunas, 
        data=modelo_selecionado.feature_importances_, 
        columns=["Importância"]
    )

    # Convertendo para porcentagem
    importancia["Importância"] *= 100  

    # Normalizando os valores
    max_importancia = importancia["Importância"].max()  
    importancia["Importância Normalizada"] = (importancia["Importância"] / max_importancia) * 100
    
     # Ordenando de forma ascendente (menor para maior)
    importancia = importancia.sort_values(by="Importância Normalizada", ascending=False)
    
    # Mantendo apenas a coluna de barra de progresso
    importancia = importancia[["Importância Normalizada"]]

    # Exibição no Streamlit
    st.dataframe(
        importancia,
        column_config={
            "Importância Normalizada": st.column_config.ProgressColumn(
                "Grau de importância (%)", format="%.2f%%", min_value=0, max_value=100
            )
        }
    )
    return importancia

importancia(modelo_selecionado, x_teste)



st.divider()
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Resultados 🎯</h4>", unsafe_allow_html=True)

def resultado():
    df_teste['profissao'] = codificador_profissao.transform(df_teste['profissao'])
    df_teste['mix_credito'] = codificador_credito.transform(df_teste['mix_credito'])
    df_teste["comportamento_pagamento"] = codificador_pagamento.transform(df_teste["comportamento_pagamento"])
    x_teste = df_teste.drop(columns=['score_credito', 'salario_anual'])
    resultados = modelo_selecionado.predict(x_teste)
    return resultados
resultados = resultado()

def exibir_resultados():
    contagem = Counter(resultados)
    categorias = []
    valores = []
    
    # Ordenar as categorias pela frequência em ordem decrescente
    for categoria, freq in sorted(contagem.items(), key=lambda item: item[1], reverse=True):
        categorias.append(categoria)
        valores.append(freq)
        st.progress(freq / sum(contagem.values()))
        st.write(f"**{categoria}:** {freq} Clientes")
        
    if st.sidebar.button('Atualizar Dados 🔄'):
        st.rerun()

exibir_resultados()

st.divider()

# === Matriz de Confusão ===
st.subheader("📌 Matriz de Confusão")

y_pred = modelo_selecionado.predict(x_teste)
cm = confusion_matrix(y_teste, y_pred, labels=modelo_selecionado.classes_)

fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo_selecionado.classes_)
disp.plot(ax=ax_cm, cmap='Blues')
st.pyplot(fig_cm)

st.divider()

# === Curva ROC (One-vs-Rest) ===
from sklearn.preprocessing import label_binarize

st.subheader("📊 Curva ROC - Multiclasse")

# Binarizando as classes
classes = modelo_selecionado.classes_
y_teste_bin = label_binarize(y_teste, classes=classes)
y_score = modelo_selecionado.predict_proba(x_teste)

fig_roc, ax_roc = plt.subplots()

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_teste_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, lw=2, label=f'Classe {classes[i]} (AUC = {roc_auc:.2f})')

ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('Taxa de Falsos Positivos')
ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
ax_roc.set_title('Curva ROC - Multiclasse')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

