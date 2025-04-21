# 📊 Projeto de Análise de Crédito com PyCaret e Streamlit

Este projeto aplica técnicas de **Machine Learning** com a biblioteca **PyCaret** e interface interativa via **Streamlit** para prever a inadimplência de clientes com base em dados socioeconômicos.

---

## 🚀 Tecnologias Utilizadas

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyCaret](https://pycaret.gitbook.io/docs/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Pandas, Numpy, Matplotlib, Seaborn](https://pandas.pydata.org/)

---

## 📂 Organização do Projeto

- `credit_score.py`: script principal com toda a lógica e interface em Streamlit
- `logo.png`: imagem utilizada no cabeçalho da aplicação
- `models/`: pasta com os modelos salvos
- `data/`: base de dados usada no projeto (amostrada de 50.000 registros)

---

## 🧠 Funcionalidades

- Análise exploratória dos dados
- Correlação entre variáveis
- Tratamento de variáveis com `setup()` do PyCaret
- Comparação automática de modelos
- Ajuste de hiperparâmetros (`tune_model`)
- Gráficos: curva AUC, matriz de confusão e importância de variáveis
- Salvamento e carregamento do melhor modelo
- Interface interativa com Streamlit

---

## 🏃‍♂️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
