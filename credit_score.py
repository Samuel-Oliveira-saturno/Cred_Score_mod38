# Importando as Bibliotecas 
from sklearn import datasets
import streamlit as st # type: ignore
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
from scipy.stats import ks_2samp
from scipy.stats import t
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

#######
st.set_page_config(page_title="Análise de Crédito", layout="wide")

# Cabeçalho com imagem hospedada
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <img src='https://github.com/Samuel-Oliveira-saturno/Cred_Score_mod38_Projeto_Final/blob/main/logo_banco.jpg' width='150'>
        <h1 style='color: #4CAF50;'>Projeto de Análise de Crédito</h1>
        <p style='font-size: 18px;'>Aplicação em PyCaret com Streamlit</p>
        <hr style='border: 1px solid #ccc;'>
    </div>
    """,
    unsafe_allow_html=True
)


#######

st.title('📊 Análise de Crédito')
# Botâo para carregar o banco de dados
uploaded_file = st.file_uploader("📁Carregar banco de Dados (.ftr)", type="ftr")

if uploaded_file is not None:
    df = pd.read_feather(uploaded_file)
    st.success("Arquivo carregado com sucesso!")

    st.subheader("📌Visualização dos dados")
    st.dataframe(df)

    # Tratamento de dados 
    st.subheader("📈 Estatísticas")
    st.write(df.describe())

    # Obter o valor mínimo e máximo da coluna 'data_ref'
    st.subheader("📅 Intervalo de datas em 'data_ref'")
    min_date = df['data_ref'].min()
    max_date = df['data_ref'].max()
    st.write(f"🔽 Data mínima: {min_date}")
    st.write(f"🔼 Data máxima: {max_date}")

     # Retornando os últimos 3 meses (a partir de 01/01/2016 como exemplo)
    st.subheader("📅 Últimos 3 meses (simulados)")
    meses = 3  # Define a quantidade de meses a serem gerados

    # Cria uma série de datas, iniciando em '01/01/2016', com 'meses' períodos e frequência mensal
    data = pd.Series(pd.date_range('1/1/2016', periods=meses, freq='MS'))

    # Converte a série em um DataFrame e renomeia a coluna
    date = pd.DataFrame(data)
    date = date.rename({0: 'oot'}, axis='columns')

    # Exibe o DataFrame resultante
    st.dataframe(date)

    st.subheader("📌 Descritiva Básica Univariada")

    # Número de linhas do DataFrame 'df'
    st.write(f"🔢 Número de linhas no DataFrame: {df.shape[0]}")

    # Contagem de ocorrências de cada valor no DataFrame 'date'
    st.write("📆 Ocorrências das datas em 'date':")
    st.dataframe(date.value_counts())

    # Contagem de ocorrências de cada valor único na coluna 'data_ref' do DataFrame 'df'
    st.write("🕒 Ocorrências por data em 'data_ref':")
    st.dataframe(df['data_ref'].value_counts())

    st.subheader("ℹ️ Informações do DataFrame (df.info())")

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)


st.subheader("📉 Distribuição da variável 'mau'")

var = 'mau'

if var in df.columns:
        st.write("Contagem de valores únicos:")
        st.dataframe(df[var].value_counts())

        # Criar gráfico de barras
        fig, ax = plt.subplots()
        df[var].value_counts().plot.bar(ax=ax, color='skyblue')
        ax.set_title(f"Distribuição da variável '{var}'")
        ax.set_xlabel(var)
        ax.set_ylabel("Frequência")

        # Exibir gráfico no Streamlit
        st.pyplot(fig)
else:
        st.warning(f"A coluna '{var}' não foi encontrada no DataFrame.")


st.subheader("📊 Distribuição da variável 'sexo'")

if 'sexo' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['sexo'].value_counts())

        # Limpar figuras anteriores
        plt.clf()

        var = "sexo"

        # Criar histograma com seaborn
        fig = sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        plt.title("Distribuição da variável 'sexo'", fontsize=14)

        # Exibir gráfico no Streamlit
        st.pyplot(fig)
else:
        st.warning("A coluna 'sexo' não foi encontrada no DataFrame.")


st.subheader("🚗 Distribuição da variável 'posse_de_veiculo'")

if 'posse_de_veiculo' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['posse_de_veiculo'].value_counts())

        # Limpar figuras anteriores
        plt.clf()

        var = "posse_de_veiculo"

        # Criar histograma com seaborn
        fig = sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        plt.title("Distribuição da variável 'posse_de_veiculo'", fontsize=14)

        # Exibir gráfico no Streamlit
        st.pyplot(fig)
else:
        st.warning("A coluna 'posse_de_veiculo' não foi encontrada no DataFrame.")


st.subheader("🏠 Distribuição da variável 'posse_de_imovel'")

if 'posse_de_imovel' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['posse_de_imovel'].value_counts())

        # Limpar figuras anteriores
        plt.clf()

        var = "posse_de_imovel"

        # Criar histograma com seaborn
        fig = sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        plt.title("Distribuição da variável 'posse_de_imovel'", fontsize=14)

        # Exibir gráfico no Streamlit
        st.pyplot(fig)
else:
        st.warning("A coluna 'posse_de_imovel' não foi encontrada no DataFrame.")



st.subheader("💼 Distribuição da variável 'tipo_renda'")

if 'tipo_renda' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['tipo_renda'].value_counts())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com seaborn
        sns.countplot(x=df["tipo_renda"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'tipo_renda'", fontsize=14)
        plt.xlabel("Tipo de Renda", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibir no Streamlit
        st.pyplot(plt.gcf())  # gcf = get current figure
else:
        st.warning("A coluna 'tipo_renda' não foi encontrada no DataFrame.")


st.subheader("🎓 Distribuição da variável 'educacao'")

if 'educacao' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['educacao'].value_counts())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com seaborn
        sns.countplot(x=df["educacao"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'educacao'", fontsize=14)
        plt.xlabel("Educação", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibir no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'educacao' não foi encontrada no DataFrame.")


st.subheader("💍 Distribuição da variável 'estado_civil'")

if 'estado_civil' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['estado_civil'].value_counts())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com seaborn
        sns.countplot(x=df["estado_civil"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'estado_civil'", fontsize=14)
        plt.xlabel("Estado Civil", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibir no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'estado_civil' não foi encontrada no DataFrame.")



st.subheader("🏠 Distribuição da variável 'tipo_residencia'")

if 'tipo_residencia' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['tipo_residencia'].value_counts())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com seaborn
        sns.countplot(x=df["tipo_residencia"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'tipo_residencia'", fontsize=14)
        plt.xlabel("Tipo de Residência", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibir no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'tipo_residencia' não foi encontrada no DataFrame.")


st.subheader("👶 Distribuição da variável 'qtd_filhos'")

if 'qtd_filhos' in df.columns:
        # Contagem dos valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['qtd_filhos'].value_counts())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com seaborn
        sns.countplot(x=df["qtd_filhos"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'qtd_filhos'", fontsize=14)
        plt.xlabel("Quantidade de Filhos", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibir no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'qtd_filhos' não foi encontrada no DataFrame.")



st.subheader("📊 Distribuição da variável 'idade'")

if 'idade' in df.columns:
        # Contagem de valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['idade'].value_counts().sort_index())

        # Define o tamanho da figura
        plt.figure(figsize=(10, 5))

        # Gráfico de barras com Seaborn
        sns.countplot(x=df["idade"], color='skyblue', edgecolor='black', alpha=0.8)

        # Títulos e rótulos
        plt.title("Distribuição da variável 'idade'", fontsize=14)
        plt.xlabel("Idade", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=90)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'idade' não foi encontrada no DataFrame.")


st.subheader("🕒 Distribuição da variável 'tempo_emprego'")

if 'tempo_emprego' in df.columns:
        # Contagem de valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['tempo_emprego'].value_counts().sort_index())

        # Limpa figura para evitar sobreposição
        plt.clf()

        # Define variável de interesse
        var = "tempo_emprego"

        # Cria histograma com Seaborn
        sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        # Exibe gráfico
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'tempo_emprego' não foi encontrada no DataFrame.")


st.subheader("🏠 Distribuição da variável 'qt_pessoas_residencia'")

if 'qt_pessoas_residencia' in df.columns:
        # Contagem de valores únicos
        st.write("Contagem de valores únicos:")
        st.dataframe(df['qt_pessoas_residencia'].value_counts().sort_index())

        # Limpa figura para evitar sobreposição
        plt.clf()

        # Define variável
        var = "qt_pessoas_residencia"

        # Cria histograma com Seaborn
        sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        # Exibe gráfico
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'qt_pessoas_residencia' não foi encontrada no DataFrame.")


st.subheader("💰 Distribuição da variável 'renda'")

if 'renda' in df.columns:
        # Contagem de valores únicos (opcional)
        st.write("Estatísticas descritivas da variável 'renda':")
        st.dataframe(df['renda'].describe())

        # Limpa figura para evitar sobreposição
        plt.clf()

        # Define variável
        var = "renda"

        # Cria histograma com Seaborn
        sns.displot(df, x=var, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

        # Exibe gráfico
        st.pyplot(plt.gcf())
else:
        st.warning("A coluna 'renda' não foi encontrada no DataFrame.")


st.subheader("✅ Análise Bivariada: Variável categórica vs 'mau'")

    # Seleciona variáveis categóricas automaticamente (excluindo a 'mau')
variaveis_categoricas = df.select_dtypes(include='object').columns.tolist()

    # Adiciona manualmente outras que sejam categóricas mas não de tipo 'object'
outras_categoricas = ['sexo', 'posse_de_imovel', 'posse_de_veiculo', 'tipo_renda', 
                          'educacao', 'estado_civil', 'tipo_residencia', 'qtd_filhos']

for var in outras_categoricas:
        if var not in variaveis_categoricas and var in df.columns:
            variaveis_categoricas.append(var)

    # Interface interativa para escolher variável categórica
var_categorica = st.selectbox("Selecione uma variável categórica:", variaveis_categoricas)

if var_categorica in df.columns and 'mau' in df.columns:
        plt.clf()
        sns.barplot(x=var_categorica, y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Personaliza o gráfico
        plt.title(f"Média de 'mau' por categoria de '{var_categorica}'", fontsize=14)
        plt.xlabel(var_categorica, fontsize=12)
        plt.ylabel("Média de 'mau'", fontsize=12)
        plt.xticks(rotation=45)

        # Exibe gráfico
        st.pyplot(plt.gcf())
else:
        st.warning("Coluna 'mau' ou variável categórica não encontrada.")


st.subheader("Análise bivariada: posse de veículo vs inadimplência")

# Limpa a figura anterior (caso esteja em loop ou repetição)
plt.clf()

# Gráfico de barras da média de 'mau' para cada valor de 'posse_de_veiculo'
sns.barplot(x='posse_de_veiculo', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

# Personalização do gráfico
plt.title("Inadimplência média por posse de veículo", fontsize=14)
plt.xlabel("Possui veículo", fontsize=12)
plt.ylabel("Média de inadimplência (mau)", fontsize=12)

# Exibe o gráfico
st.pyplot(plt.gcf())


st.subheader("Análise bivariada: posse de imóvel vs inadimplência")

if 'posse_de_imovel' in df.columns and 'mau' in df.columns:
        # Limpa a figura anterior
        plt.clf()

        # Cria o gráfico de barras
        sns.barplot(x='posse_de_imovel', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Personaliza
        plt.title("Inadimplência média por posse de imóvel", fontsize=14)
        plt.xlabel("Possui imóvel", fontsize=12)
        plt.ylabel("Média de inadimplência (mau)", fontsize=12)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("Colunas 'posse_de_imovel' ou 'mau' não encontradas no DataFrame.")


st.subheader("💼 Análise bivariada: tipo de renda vs inadimplência")

if 'tipo_renda' in df.columns and 'mau' in df.columns:
        # Limpa figuras anteriores
        plt.clf()

        # Cria o gráfico de barras
        tipo_renda = sns.barplot(x='tipo_renda', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Personalização do eixo X
        tipo_renda.set_xticklabels(tipo_renda.get_xticklabels(), rotation=45, horizontalalignment='right')

        # Títulos
        plt.title("Inadimplência média por tipo de renda", fontsize=14)
        plt.xlabel("Tipo de Renda", fontsize=12)
        plt.ylabel("Média de Inadimplência (mau)", fontsize=12)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("Colunas 'tipo_renda' ou 'mau' não encontradas no DataFrame.")


st.subheader("🎓 Análise bivariada: educação vs inadimplência")

if 'educacao' in df.columns and 'mau' in df.columns:
        # Limpa figuras anteriores
        plt.clf()

        # Cria o gráfico de barras
        educacao = sns.barplot(x='educacao', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Ajusta os rótulos do eixo X
        educacao.set_xticklabels(educacao.get_xticklabels(), rotation=45, horizontalalignment='right')

        # Títulos
        plt.title("Inadimplência média por nível de educação", fontsize=14)
        plt.xlabel("Educação", fontsize=12)
        plt.ylabel("Média de inadimplência (mau)", fontsize=12)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("Colunas 'educacao' ou 'mau' não encontradas no DataFrame.")


st.subheader("💍 Análise bivariada: estado civil vs inadimplência")

if 'estado_civil' in df.columns and 'mau' in df.columns:
        # Limpa figuras anteriores
        plt.clf()

        # Cria o gráfico de barras
        estado_civil = sns.barplot(x='estado_civil', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Ajusta os rótulos do eixo X
        estado_civil.set_xticklabels(estado_civil.get_xticklabels(), rotation=45, horizontalalignment='right')

        # Títulos
        plt.title("Inadimplência média por estado civil", fontsize=14)
        plt.xlabel("Estado Civil", fontsize=12)
        plt.ylabel("Média de inadimplência (mau)", fontsize=12)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("Colunas 'estado_civil' ou 'mau' não encontradas no DataFrame.")


st.subheader("🏠 Análise bivariada: tipo de residência vs inadimplência")

if 'tipo_residencia' in df.columns and 'mau' in df.columns:
        # Limpa figuras anteriores
        plt.clf()

        # Cria o gráfico de barras
        tipo_residencia = sns.barplot(x='tipo_residencia', y='mau', data=df, ci=None, palette='pastel', edgecolor='black')

        # Ajusta os rótulos do eixo X
        tipo_residencia.set_xticklabels(tipo_residencia.get_xticklabels(), rotation=90, horizontalalignment='right')

        # Títulos
        plt.title("Inadimplência média por tipo de residência", fontsize=14)
        plt.xlabel("Tipo de Residência", fontsize=12)
        plt.ylabel("Média de inadimplência (mau)", fontsize=12)

        # Exibe no Streamlit
        st.pyplot(plt.gcf())
else:
        st.warning("Colunas 'tipo_residencia' ou 'mau' não encontradas no DataFrame.")


st.subheader("🧼 Verificação de valores ausentes (missing values)")

missing_values = df.isna().sum()
st.dataframe(missing_values[missing_values > 0].sort_values(ascending=False))

if missing_values.sum() == 0:
        st.success("✅ Nenhum valor ausente encontrado no DataFrame!")


st.subheader("📌 Tratamento de valores ausentes na coluna 'tempo_emprego'")

if 'tempo_emprego' in df.columns:
        # Antes do preenchimento
        missing_before = df['tempo_emprego'].isna().sum()
        st.write(f"Valores ausentes antes do preenchimento: {missing_before}")

        # Preenchimento com a média
        df['tempo_emprego'] = df['tempo_emprego'].fillna(df['tempo_emprego'].mean())

        # Após o preenchimento
        missing_after = df['tempo_emprego'].isna().sum()
        st.write(f"Valores ausentes após o preenchimento: {missing_after}")
else:
        st.warning("Coluna 'tempo_emprego' não encontrada no DataFrame.")


st.subheader("📊 Metadados do DataFrame")

    # Cria o DataFrame com tipo de dado
metadados = pd.DataFrame(df.dtypes, columns=['dtype'])

    # Adiciona a quantidade de valores únicos por coluna
metadados['valores_unicos'] = df.nunique()

    # Exibe a tabela no Streamlit
st.dataframe(metadados)


st.subheader("🔄 Conversão da variável 'mau' para tipo inteiro")

if 'mau' in df.columns:
        df['mau'] = df['mau'].astype('int64')
        st.success("Coluna 'mau' convertida com sucesso para int64!")
        st.write(df['mau'].dtypes)
else:
        st.warning("A coluna 'mau' não foi encontrada no DataFrame.")



st.subheader("📈 Cálculo do IV (Information Value)")

    # Define a função IV
def IV(variavel, resposta):
        tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')
        rótulo_evento = tab.columns[0]
        rótulo_nao_evento = tab.columns[1]

        tab['pct_evento'] = tab[rótulo_evento] / tab.loc['total', rótulo_evento]
        tab['pct_nao_evento'] = tab[rótulo_nao_evento] / tab.loc['total', rótulo_nao_evento]
        tab['woe'] = np.log(tab.pct_evento / tab.pct_nao_evento)
        tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento) * tab.woe

        return tab['iv_parcial'].sum()

    # Seleciona variáveis categóricas para análise de IV
variaveis_categoricas = df.select_dtypes(include='object').columns.tolist()
extras = ['sexo', 'posse_de_imovel', 'posse_de_veiculo', 'tipo_renda', 
              'educacao', 'estado_civil', 'tipo_residencia']

for col in extras:
        if col not in variaveis_categoricas and col in df.columns:
            variaveis_categoricas.append(col)

    # Interface para escolha da variável
var_iv = st.selectbox("Selecione a variável para calcular o IV:", variaveis_categoricas)

if var_iv and 'mau' in df.columns:
        iv_valor = IV(df[var_iv], df['mau'])
        st.write(f"📊 IV para a variável **{var_iv}**: `{iv_valor:.4f}`")

        # Interpretação (opcional)
        if iv_valor < 0.02:
            interpret = "Variável não preditiva"
        elif iv_valor < 0.1:
            interpret = "Poder preditivo fraco"
        elif iv_valor < 0.3:
            interpret = "Poder preditivo médio"
        elif iv_valor < 0.5:
            interpret = "Boa variável preditiva"
        else:
            interpret = "Forte poder preditivo"

        st.info(f"🧠 Interpretação: **{interpret}**")
else:
        st.warning("Selecione uma variável válida e certifique-se de que 'mau' está no DataFrame.")


st.subheader("📌 IV da variável 'sexo'")

if 'sexo' in df.columns and 'mau' in df.columns:
        iv_sexo = IV(df['sexo'], df['mau'])
        st.success(f"IV da variável **SEXO**: `{iv_sexo:.1%}`")
else:
        st.warning("Colunas 'sexo' ou 'mau' não encontradas no DataFrame.")


st.subheader("📑 Construção de metadados para modelagem")

    # Cria o DataFrame com tipos e valores únicos
metadados = pd.DataFrame(df.dtypes, columns=['dtype'])
metadados['valores_unicos'] = df.nunique()
metadados['variavel'] = 'covariavel'  # define tudo inicialmente como covariável

    # Define 'mau' como variável resposta (se existir)
if 'mau' in df.columns:
        metadados.loc['mau', 'variavel'] = 'resposta'

    # Verifica se 'bom' existe antes de definir como resposta
if 'bom' in df.columns:
        metadados.loc['bom', 'variavel'] = 'resposta'

    # Exibe o resultado
st.dataframe(metadados)


st.subheader("📊 IV da variável numérica 'idade' (quintis)")

var = 'idade'

if var in df.columns and 'mau' in df.columns:
        try:
            # Divide 'idade' em 5 grupos com base nos quantis
            idade_quantis = pd.qcut(df[var], 5, duplicates='drop')

            # Calcula o IV usando a função definida anteriormente
            iv_idade = IV(idade_quantis, df['mau'])

            st.success(f"IV da variável 'idade' (dividida em quintis): `{iv_idade:.4f}`")
        except Exception as e:
            st.error(f"Erro ao calcular IV para 'idade': {e}")
else:
        st.warning("As colunas 'idade' ou 'mau' não foram encontradas no DataFrame.")



st.subheader("🧠 Cálculo do Information Value (IV) para todas as covariáveis")

    # Garante que o campo 'IV' existe
metadados['IV'] = np.nan

    # Loop pelas variáveis definidas como 'covariavel'
for var in metadados[metadados['variavel'] == 'covariavel'].index:
        try:
            if metadados.loc[var, 'valores_unicos'] > 6:
                # Trata como variável contínua (binning em quintis)
                metadados.loc[var, 'IV'] = IV(pd.qcut(df[var], 5, duplicates='drop'), df['mau'])
            else:
                # Trata como categórica
                metadados.loc[var, 'IV'] = IV(df[var], df['mau'])
        except Exception as e:
            st.warning(f"⚠️ Erro ao calcular IV para '{var}': {e}")

    # Exibe os resultados ordenados pelo IV
metadados_iv = metadados.dropna(subset=['IV']).sort_values('IV', ascending=False)
st.dataframe(metadados_iv)





def biv_discreta(var, df):
        df = df.copy()

        # Cria a variável 'bom' como complemento de 'mau'
        df['bom'] = 1 - df['mau']

        # Agrupa os dados
        g = df.groupby(var)
        biv = pd.DataFrame({
            'qt_bom': g['bom'].sum(),
            'qt_mau': g['mau'].sum(),
            'mau': g['mau'].mean(),
            'cont': g[var].count()
        })

        # Erro padrão da média de 'mau'
        biv['ep'] = np.sqrt(biv['mau'] * (1 - biv['mau']) / biv['cont'])

        # Intervalos de confiança (95%)
        biv['mau_sup'] = biv['mau'] + t.ppf(0.975, biv['cont'] - 1) * biv['ep']
        biv['mau_inf'] = biv['mau'] + t.ppf(0.025, biv['cont'] - 1) * biv['ep']

        # Logits
        biv['logit'] = np.log(biv['mau'] / (1 - biv['mau']))
        biv['logit_sup'] = np.log(biv['mau_sup'] / (1 - biv['mau_sup']))
        biv['logit_inf'] = np.log(biv['mau_inf'] / (1 - biv['mau_inf']))

        # WOE
        woe_geral = np.log(df['mau'].mean() / (1 - df['mau'].mean()))
        biv['woe'] = biv['logit'] - woe_geral
        biv['woe_sup'] = biv['logit_sup'] - woe_geral
        biv['woe_inf'] = biv['logit_inf'] - woe_geral

        # Gráficos
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        x_labels = biv.index.astype(str)

        # Gráfico WOE
        ax[0].plot(x_labels, biv['woe'], 'o-b', label='WOE')
        ax[0].plot(x_labels, biv['woe_sup'], 'o:r', label='Limite Superior')
        ax[0].plot(x_labels, biv['woe_inf'], 'o:r', label='Limite Inferior')
        ax[0].set_ylabel("Weight of Evidence")
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[0].set_xticks(range(len(x_labels)))
        ax[0].set_xticklabels(x_labels, rotation=15)

        # Gráfico de contagem
        biv['cont'].plot(kind='bar', ax=ax[1], color='lightblue', edgecolor='black')
        ax[1].set_ylabel("Contagem")
        ax[1].set_xlabel(var)
        ax[1].set_xticks(range(len(x_labels)))
        ax[1].set_xticklabels(x_labels, rotation=15)

        plt.tight_layout()
        st.pyplot(fig)

        return biv


st.subheader("💍 Análise Bivariada: estado_civil vs mau")

if 'estado_civil' in df.columns and 'mau' in df.columns:
        resultado_woe = biv_discreta('estado_civil', df)
        st.dataframe(resultado_woe)
else:
        st.warning("A coluna 'estado_civil' ou 'mau' não foi encontrada.")


st.subheader("🧪 Análise Bivariada com Agrupamento: tipo_renda vs mau")

if 'tipo_renda' in df.columns:
        # Cria uma cópia para preservar o DataFrame original
        df2 = df.copy()

        # Agrupamento de categorias
        df2['tipo_renda'] = df2['tipo_renda'].replace({
            'Bolsista': 'Bols./SerPubl',
            'Servidor público': 'Bols./SerPubl'
        })

        # Aplica a análise bivariada com WOE
        resultado_tipo_renda = biv_discreta('tipo_renda', df2)
        st.dataframe(resultado_tipo_renda)
else:
        st.warning("A coluna 'tipo_renda' não foi encontrada.")



st.subheader("📈 IV da variável 'tipo_renda' após agrupamento")

if 'tipo_renda' in df.columns and 'mau' in df.columns:
        # Cria cópia modificada com categorias agrupadas
        df2 = df.copy()
        df2['tipo_renda'] = df2['tipo_renda'].replace({
            'Bolsista': 'Bols./SerPubl',
            'Servidor público': 'Bols./SerPubl'
        })

        # Calcula o IV da nova versão da variável
        iv_tipo_renda_agrupada = IV(df2['tipo_renda'], df['mau'])

        st.success(f"IV de 'tipo_renda' após agrupamento: `{iv_tipo_renda_agrupada:.4f}`")
else:
        st.warning("Colunas 'tipo_renda' ou 'mau' não encontradas.")


st.subheader("🎓 Análise Bivariada com Agrupamento: educacao vs mau")

if 'educacao' in df.columns:
        # Cópia do DataFrame para não alterar o original
        df2 = df.copy()

        # Agrupamento das categorias de escolaridade
        df2['educacao'] = df2['educacao'].replace({
            'Superior completo': 'Sup.Compl/PosGra',
            'Pós graduação': 'Sup.Compl/PosGra'
        })

        # Aplica análise bivariada com WOE
        resultado_educacao = biv_discreta('educacao', df2)
        st.dataframe(resultado_educacao)
else:
        st.warning("A coluna 'educacao' não foi encontrada.")


iv_educacao_agrupada = IV(df2['educacao'], df['mau'])
st.success(f"IV de 'educacao' após agrupamento: `{iv_educacao_agrupada:.4f}`")


st.subheader("🎓 Análise Bivariada com Agrupamento Refinado: educacao vs mau")

if 'educacao' in df.columns:
        # Cópia do DataFrame
        df2 = df.copy()

        # Primeiro agrupamento anterior
        df2['educacao'] = df2['educacao'].replace({
            'Superior completo': 'Sup.Compl/PosGra',
            'Pós graduação': 'Sup.Compl/PosGra'
        })

        # Agrupamento refinado adicional
        df2['educacao'] = df2['educacao'].replace({
            'Superior incompleto': 'Sup.Compl/PosGra',
            'Fundamental': 'Funda./Méd',
            'Médio': 'Funda./Méd'
        })

        # Aplica análise bivariada
        resultado_educacao = biv_discreta('educacao', df2)
        st.dataframe(resultado_educacao)

        # Calcula o novo IV após o agrupamento
        iv_educacao_agrupada = IV(df2['educacao'], df['mau'])
        st.success(f"IV de 'educacao' após agrupamento completo: `{iv_educacao_agrupada:.4f}`")
else:
        st.warning("A coluna 'educacao' não foi encontrada.")


st.subheader("📅 Crosstab entre 'mau' e base out-of-time (oot)")

try:
        crosstab = pd.crosstab(df['mau'], date['oot'])
        st.dataframe(crosstab)
except Exception as e:
        st.error(f"Erro ao gerar crosstab: {e}")


st.subheader("🔍 Verificação de valores ausentes por coluna")
st.dataframe(df.isna().sum()[df.isna().sum() > 0])


#teste

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Identifica colunas numéricas e categóricas
num_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=['mau']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Pipelines individuais
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Une os pipelines com ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Pipeline completo com modelo
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])


# Supondo que a variável alvo é 'mau'
X = df.drop(columns=['mau'])
y = df['mau']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento
model_pipeline.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Predições e probabilidades
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Acurácia
acc = accuracy_score(y_test, y_pred)

# AUC (área sob a curva ROC)
auc = roc_auc_score(y_test, y_proba)

# Gini = 2*AUC - 1
gini = 2 * auc - 1

# KS
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
ks = max(tpr - fpr)

print(f"Acurácia: {acc:.3f}")
print(f"AUC: {auc:.3f}")
print(f"Gini: {gini:.3f}")
print(f"KS: {ks:.3f}")

#st.subheader("📋 Primeiras linhas do conjunto de dados (X_digits)")

#st.dataframe(X_digits.head())

from sklearn.datasets import load_digits

# Carrega os dados de exemplo
X_digits, y_digits = load_digits(return_X_y=True, as_frame=True)


st.subheader("Primeiras linhas do conjunto de dados (X_digits)")
st.dataframe(X_digits.head())

st.subheader("📋 Primeiras linhas do DataFrame (df2) com pré-processamento")

st.dataframe(df2.head())



#####
# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Título do app
st.title("🔢 PCA + Regressão Logística com Digits")

# Carrega os dados
X_digits, y_digits = load_digits(return_X_y=True, as_frame=True)
st.write("Formato do conjunto de dados:", X_digits.shape)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.3, random_state=10
)

# Sidebar para configurar número de componentes principais
n_components = st.sidebar.slider("Componentes do PCA", 2, X_train.shape[1], 20)

# Pipeline com scaler, PCA e regressão logística
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('logreg', LogisticRegression(max_iter=200))
])

# Treinamento do modelo
pipeline.fit(X_train, y_train)

# Previsões
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Métricas
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score((y_test == y_test.max()).astype(int), y_proba)  # binarização simples para ROC
gini = 2 * auc - 1
fpr, tpr, _ = roc_curve((y_test == y_test.max()).astype(int), y_proba)
ks = max(tpr - fpr)

# Exibe resultados
st.subheader("📊 Resultados")
st.write(f"**Acurácia:** {acc:.2%}")
st.write(f"**Gini:** {gini:.4f}")
st.write(f"**KS:** {ks:.4f}")

# Curva ROC
st.subheader("📉 Curva ROC")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='ROC Curve')
ax.plot([0, 1], [0, 1], 'k--', label='Aleatório')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("Curva ROC")
ax.legend()
st.pyplot(fig)

#####

#####
# Importa o StandardScaler
from sklearn.preprocessing import StandardScaler

# Título da seção
st.subheader("⚙️ Padronização dos dados (Z-score)")

# Cria uma instância do StandardScaler
scaler = StandardScaler()

# Fit no X_train
scaler.fit(X_train)

# Transforma X_train e X_test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Exibe os dados transformados
st.write("📉 Primeiras linhas do X_train com Z-score aplicado:")
st.dataframe(pd.DataFrame(X_train_scaled).head())

#####

# Importa o PCA
from sklearn.decomposition import PCA

# Título da seção
st.subheader("🔻 Aplicando PCA (Redução de Dimensionalidade)")

# Slider para escolher o número de componentes
n_components = st.slider("Número de componentes principais", 2, X_train_scaled.shape[1], 20)

# Instancia o PCA com o número de componentes definido
pca = PCA(n_components=n_components)

# Ajusta (fit) o PCA nos dados padronizados
pca.fit(X_train_scaled)

# Transforma os dados de treino
X_train_pca = pca.transform(X_train_scaled)

# Transforma os dados de teste também (com base no PCA do treino)
X_test_pca = pca.transform(X_test_scaled)

# Mostra a variância explicada acumulada
st.write("📈 Variância explicada acumulada:")
st.line_chart(np.cumsum(pca.explained_variance_ratio_))

# Mostra os dados reduzidos
st.write("🔍 Primeiras linhas após PCA:")
st.dataframe(pd.DataFrame(X_train_pca).head())

#######

# Importa a Regressão Logística
from sklearn.linear_model import LogisticRegression

# Título da seção
st.subheader("🤖 Treinando o Modelo: Regressão Logística")

# Instancia o modelo
logistic = LogisticRegression(max_iter=200)

# Treina o modelo com os dados já reduzidos pelo PCA
logistic.fit(X_train_pca, y_train)

# Faz as previsões no conjunto de treino
y_train_pred = logistic.predict(X_train_pca)

# Exibe as primeiras previsões
st.write("📋 Primeiras previsões (treino):")
st.write(y_train_pred[:10])

######

st.subheader("🧪 Previsões no conjunto de teste")

# Aplica Z-score no teste com base no treino
X_test_scaled = scaler.transform(X_test)

# Aplica PCA no teste com base no treino
X_test_pca = pca.transform(X_test_scaled)

# Faz previsões no conjunto de teste
y_test_pred = logistic.predict(X_test_pca)
y_test_proba = logistic.predict_proba(X_test_pca)[:, 1]

# Exibe as primeiras previsões
st.write("📋 Previsões (primeiras 10):", y_test_pred[:10])


#######

# Importações necessárias
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

st.subheader("🔗 Pipeline completo: Z-score + PCA + Regressão Logística")

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.3, random_state=10
)

# Define os componentes do pipeline
scaler_pipe = StandardScaler()
pca_pipe = PCA(n_components=15)
logistic_pipe = LogisticRegression(max_iter=200)

# Monta o pipeline
pipeline = Pipeline([
    ('scaler', scaler_pipe),
    ('pca', pca_pipe),
    ('logreg', logistic_pipe)
])

# Treina o pipeline
pipeline.fit(X_train, y_train)

# Previsões no teste
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

# Exibe as primeiras previsões
st.write("📋 Previsões do pipeline (primeiras 10):", y_test_pred[:10])


#####

st.subheader("⚙️ Montando pipeline final: pré-processamento + modelo")

# Define o pipeline com os passos nomeados
pipe = Pipeline(steps=[
    ("scaler", scaler_pipe),     # Padronização (Z-score)
    ("pca", pca_pipe),           # Redução de dimensionalidade
    ("logistic", logistic_pipe)  # Modelo de regressão
])

# Treina o pipeline com os dados de treino
pipe.fit(X_train, y_train)

# Faz previsões no teste
y_test_pred = pipe.predict(X_test)
y_test_proba = pipe.predict_proba(X_test)[:, 1]

# Exibe primeiras previsões
st.write("📋 Previsões do pipeline (primeiras 10):")
st.write(y_test_pred[:10])

#########

st.subheader("🔍 Etapas do Pipeline")

# Exibe os nomes e objetos das etapas do pipeline
st.write("📦 Componentes do pipeline:")
for nome, etapa in pipe.named_steps.items():
    st.markdown(f"**{nome}**: `{etapa}`")

######
st.subheader("🏋️‍♂️ Treinamento do Pipeline")

# Treinando o pipeline
pipe.fit(X_train, y_train)

st.success("✅ Pipeline treinado com sucesso nos dados de treino!")


st.subheader("🧠 Previsões com Pipeline Treinado")

# Previsões no conjunto de treino
y_train_pred = pipe.predict(X_train)
st.write("📋 Previsões no treino (primeiras 10):")
st.write(y_train_pred[:10])

# Previsões no conjunto de teste
y_test_pred = pipe.predict(X_test)
st.write("📋 Previsões no teste (primeiras 10):")
st.write(y_test_pred[:10])

# Exibe as etapas do pipeline
st.subheader("🔧 Etapas do Pipeline")
st.write("📦 named_steps:")
st.json({nome: str(etapa) for nome, etapa in pipe.named_steps.items()})

st.write("📦 steps (tuplas):")
st.write(pipe.steps)


st.subheader("🔍 Componentes principais (PCA) extraídos do pipeline")

try:
    X_train_pca = pipe.named_steps['pca'].transform(X_train)
    st.write("✅ X_train_pca (primeiras linhas):")
    st.dataframe(pd.DataFrame(X_train_pca).head())
except Exception as e:
    st.error(f"Erro ao aplicar PCA extraído do pipeline: {e}")


st.subheader("🧱 Codificação de variáveis categóricas (dummies)")

if 'df' in locals():
    dummy = pd.get_dummies(df,
        columns=['sexo', 'posse_de_veiculo', 'posse_de_imovel',
                 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'])
    st.write("✅ DataFrame com variáveis dummies:")
    st.dataframe(dummy.head())
else:
    st.warning("⚠️ DataFrame 'df' não foi carregado.")


import streamlit as st
import pandas as pd

# Carrega o dataframe original
# 

# Apenas para exemplo, vamos criar um DataFrame dummy se não tiver um ainda
# df = pd.DataFrame({...})

st.title("Pré-processamento de Dados para Modelagem")

# Exibe os tipos de dados do dummy (se existir)
if 'dummy' in locals():
    st.subheader("Tipos de dados (dummy):")
    st.write(dummy.dtypes)

# Amostragem inicial
dataset = df.sample(40000, random_state=42)

st.subheader("Colunas do DataFrame:")
st.write(df.columns.tolist())

# Remove colunas desnecessárias
cols_to_drop = ['data_ref', 'index']
dataset.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')  # evita erro se a coluna não existir

# Divide o dataset
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)

# Reseta índices
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# Exibe os tamanhos dos conjuntos
st.subheader("Tamanhos dos conjuntos de dados:")
st.write(f"Conjunto para modelagem (treino/teste): {data.shape}")
st.write(f"Conjunto não visto (validação): {data_unseen.shape}")


########################

from pycaret.classification import *

from pycaret.classification import ClassificationExperiment

exp = ClassificationExperiment()

exp.setup(
    data=data,
    target='mau',
    experiment_name='credit_1',
    normalize=True,
    normalize_method='zscore',
    transformation=True,
    transformation_method='quantile',
    fix_imbalance=True
)


best_model = exp.compare_models(fold=4, sort='AUC')
exp.plot_model(best_model, plot='feature')
exp.save_model(best_model, 'LR_Model_Aula5_062022')

###############################

#Modificando os dados de treinamento 

from pycaret.classification import*

# Simulação do carregamento dos dados 

st.title("Pré-processamento de Dados")

# Exibe os tipos das colunas antes da conversão
st.subheader("Tipos de dados antes da conversão:")
st.write(data.dtypes)

# Conversão da coluna 'qtd_filhos' para float
if 'qtd_filhos' in data.columns:
    try:
        data['qtd_filhos'] = data['qtd_filhos'].astype(float)
        st.success("Coluna 'qtd_filhos' convertida para float com sucesso!")
    except Exception as e:
        st.error(f"Erro ao converter 'qtd_filhos': {e}")
else:
    st.warning("Coluna 'qtd_filhos' não encontrada no DataFrame.")

# Tipos de dados após a conversão
st.subheader("Tipos de dados após a conversão:")
st.write(data.dtypes)

########

st.title("Análise Exploratória")

# Supondo que o DataFrame `data` já está carregado e pré-processado

# 1. Correlação entre variáveis numéricas
st.subheader("Matriz de Correlação")

numeric_data = data.select_dtypes(include=['number'])
data_corr = numeric_data.corr()
st.dataframe(data_corr)

# Heatmap da correlação (opcional)
st.subheader("Heatmap de Correlação")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
st.pyplot(fig)

# 2. Distribuição percentual da variável alvo 'mau'
st.subheader("Distribuição da Variável Alvo ('mau')")

if 'mau' in data.columns:
    target_distribution = data['mau'].value_counts(normalize=True).rename_axis('Classe').reset_index(name='Proporção')
    st.dataframe(target_distribution)
    st.bar_chart(target_distribution.set_index('Classe'))
else:
    st.warning("Coluna 'mau' não encontrada no DataFrame.")

########
from pycaret.classification import ClassificationExperiment

st.subheader("Configuração do Experimento com PyCaret")

# Cria o experimento
exp = ClassificationExperiment()

# Setup do experimento
with st.spinner("Configurando o experimento..."):
    try:
        exp.setup(
            data=data,
            target='mau',
            experiment_name='credit_1',
            normalize=True,
            normalize_method='zscore',
            transformation=True,
            transformation_method='quantile',
            fix_imbalance=True,
            verbose=False,  # evita prints grandes no terminal
        )
        st.success("Experimento configurado com sucesso!")
    except Exception as e:
        st.error(f"Ocorreu um erro no setup do experimento: {e}")

# Continuando após o setup...
st.subheader("Comparação de Modelos")

# Botão para comparar modelos
if st.button("Encontrar Melhor Modelo (AUC)"):
    with st.spinner("Comparando modelos... Isso pode levar alguns segundos."):
        try:
            best_model = exp.compare_models(fold=4, sort='AUC')
            st.success(f"Melhor modelo encontrado: {type(best_model).__name__}")
            st.write(best_model)
        except Exception as e:
            st.error(f"Ocorreu um erro ao comparar os modelos: {e}")
else:
    best_model = None

# Plots só se o melhor modelo já foi gerado
if best_model is not None:
    st.subheader("Importância das Variáveis")
    with st.spinner("Gerando gráfico de importância..."):
        exp.plot_model(best_model, plot='feature', display_format='streamlit')

    st.subheader("Curva ROC / AUC")
    with st.spinner("Gerando curva AUC..."):
        exp.plot_model(best_model, plot='auc', display_format='streamlit')


# Salvando o modelo

from pycaret.classification import save_model, load_model
import os

st.subheader("Salvar e Carregar Modelo")

# Caminho/nome do modelo
model_name = 'LR_Model_Aula_5_062022'

# Botão para salvar modelo
if best_model is not None:
    if st.button("Salvar Modelo"):
        try:
            save_model(best_model, model_name)
            st.success(f"Modelo salvo com o nome: {model_name}")
        except Exception as e:
            st.error(f"Erro ao salvar modelo: {e}")

# Botão para carregar modelo
if os.path.exists(model_name + '.pkl'):
    if st.button("Carregar Modelo"):
        try:
            model_loaded = load_model(model_name)
            st.success("Modelo carregado com sucesso!")
            st.write("Modelo carregado:", type(model_loaded).__name__)

            # Exibir componentes do pipeline, se aplicável
            if hasattr(model_loaded, 'named_steps'):
                st.subheader("Componentes do Pipeline:")
                st.write(model_loaded.named_steps)
            else:
                st.info("O modelo carregado não é um pipeline com named_steps.")
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
else:
    st.warning("Modelo ainda não foi salvo. Clique em 'Salvar Modelo' primeiro.")


st.subheader("Novo Setup com Dataset 'dummy'")

exp2 = ClassificationExperiment()

# Setup com tratamento de multicolinearidade e binning
with st.spinner("Configurando o novo experimento..."):
    try:
        exp2.setup(
            data=dummy,
            target='mau',
            normalize=True,
            transformation=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            bin_numeric_features=[
                'qtd_filhos', 'idade', 'tempo_emprego', 
                'qt_pessoas_residencia', 'renda'
            ],
            verbose=False
        )
        st.success("Novo experimento configurado com sucesso usando 'dummy'!")
    except Exception as e:
        st.error(f"Ocorreu um erro no novo setup: {e}")


st.subheader("Criação e Avaliação do Modelo LightGBM")

# Botão para criar o modelo LightGBM
if st.button("Criar e Avaliar LightGBM"):
    with st.spinner("Criando modelo LightGBM..."):
        try:
            lightgbm = exp2.create_model('lightgbm')
            st.success("Modelo LightGBM criado com sucesso!")
            st.write(lightgbm)

            st.info("Ajustando hiperparâmetros...")
            tuned_lightgbm = exp2.tune_model(lightgbm)
            st.success("Modelo LightGBM tunado com sucesso!")
            st.write(tuned_lightgbm)

            final_lightgbm = exp2.finalize_model(tuned_lightgbm)
            st.success("Modelo finalizado e pronto para produção!")

            st.subheader("Avaliação do Modelo Final")
            exp2.evaluate_model(final_lightgbm)

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processo com LightGBM: {e}")


st.subheader("Gráficos do Modelo Final - LightGBM")

# Botão para exibir os gráficos
if st.button("Exibir Gráficos (AUC e Confusion Matrix)"):
    with st.spinner("Gerando curva AUC..."):
        try:
            exp2.plot_model(final_lightgbm, plot='auc', display_format='streamlit')
            st.success("Curva AUC exibida com sucesso!")
        except Exception as e:
            st.error(f"Erro ao exibir curva AUC: {e}")

    with st.spinner("Gerando Matriz de Confusão..."):
        try:
            exp2.plot_model(final_lightgbm, plot='confusion_matrix', display_format='streamlit')
            st.success("Matriz de Confusão exibida com sucesso!")
        except Exception as e:
            st.error(f"Erro ao exibir matriz de confusão: {e}")
