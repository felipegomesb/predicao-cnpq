# predicao-cnpq

Modelos de aprendizado de máquina para predição do valor de bolsas e auxílios concedidos pelo CNPq, utilizando 21 anos de dados abertos governamentais (2003–2025). Foram avaliados Ridge, Random Forest, LightGBM, XGBoost e TabPFN-3, com o TabPFN-3 obtendo o melhor desempenho (R² = 0,634, MAE = R$ 2.998,53).

## Setup

Pré-requisito: Python 3.10+

```bash
# Clone o repositório
git clone https://github.com/felipegomesb/predicao-cnpq
cd predicao-cnpq

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

## TabPFN

O TabPFN-3 é executado via API em servidores com GPU. É necessário criar uma conta em [https://priorlabs.ai/tabpfn](https://priorlabs.ai/tabpfn) e obter uma chave de acesso.

```bash
# Autentique com sua chave
python -c "import tabpfn_client; tabpfn_client.set_access_token('SUA_KEY_AQUI')"
```

A partir disso o `tabpfn_client` já está autenticado para a sessão. A chave fica salva localmente e não precisa ser informada novamente nas próximas execuções.
