# 🏥 HealthSearch: Motor de Busca Híbrido Clínico

Projeto desenvolvido para o **Laboratório Prático 05 - Desafio Integrador** da disciplina de **Tendências em Ciência da Computação (Recuperação de Informação / Processamento de Linguagem Natural)** do Centro Universitário de João Pessoa (**UNIPÊ**), sob orientação do **Prof. Me. Ricardo Roberto de Lima**.

---

## 👥 Equipe de Desenvolvimento

- Júlio César Carvalho Santos — Desenvolvimento e documentação
- Perilo Oliveira Viana Sobrinho — Desenvolvimento

---

## 📌 Visão Geral do Problema

Sistemas hospitalares enfrentam dilemas críticos de recuperação da informação médica:

- **Falha Léxica Pura:** Consultas como *"ataque cardíaco"* não encontram diretrizes cadastradas estritamente sob termos técnicos formais como *"síndrome coronariana aguda"* ou *"isquemia miocárdica"*.
- **Falha Semântica Pura:** Códigos de exames específicos (ex: `CÓD-ECG-12D`) ou dosagens têm sua precisão diluída em espaços vetoriais, trazendo apenas documentos genéricos sobre cardiologia.

O **HealthSearch** soluciona esses pontos cegos integrando o algoritmo léxico **Okapi BM25** e representações densas vetoriais (**Sentence-Transformers**) por meio do algoritmo **Reciprocal Rank Fusion (RRF)**, com reordenação opcional via **Cross-Encoder**.

---

## 🗂️ Estrutura do Repositório

```
healthSearch/
├── healthsearch_app.py   # Aplicação Streamlit (arquivo único)
├── requirements.txt      # Dependências do projeto
├── README.md
└── .gitignore
```

---

## 🏗️ Arquitetura do Sistema

A aplicação é implementada em **um único arquivo** (`healthsearch_app.py`), organizado em fases sequenciais:

1. **Fase 1 — Ingestão e Pré-processamento:** Corpus médico **hardcoded** com 6 protocolos clínicos. Normalização para minúsculas, remoção de acentos, tokenização preservando códigos alfanuméricos com hífen (ex: `CÓD-ECG-12D` → `cod-ecg-12d`) e eliminação de stopwords em português **exclusivamente** no motor léxico BM25.
2. **Fase 2 — Motor Léxico (BM25):** Indexação com `BM25Okapi` (`rank-bm25`) sobre o corpus tokenizado. Hiperparâmetros controlados por sliders: $k_1$ (0.0–3.0, padrão `1.2`) e $b$ (0.0–1.0, padrão `0.75`). Gera o ranking ordinal $1$ a $N$.
3. **Fase 3 — Motor Semântico Vetorial:** Embeddings densos do modelo multilíngue `paraphrase-multilingual-MiniLM-L12-v2` (`sentence-transformers`) e **Similaridade de Cosseno** entre a consulta e os documentos (via `util.cos_sim`). A entrada do Bi-Encoder **não** sofre remoção de stopwords, preservando a sintaxe das sentenças.
4. **Fase 4 — Fusão Híbrida RRF:** Combina os ranks ordinais dos dois motores (nunca scores brutos):

   $$\text{Score}_{RRF}(D) = \alpha \cdot \left[ \frac{1}{k_{rrf} + \text{Rank}_{BM25}(D)} \right] + (1 - \alpha) \cdot \left[ \frac{1}{k_{rrf} + \text{Rank}_{Semantico}(D)} \right]$$

   - $\alpha$ (peso léxico): slider de `0.0` a `1.0` (padrão `0.5`; `1.0` = 100% léxico, `0.0` = 100% semântico).
   - $k_{rrf}$ (constante de suavização): **fixo em `60`** (exibido na Sidebar).
5. **Módulo Bônus — Cross-Encoder Re-Ranking:** Modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` aplicado **somente ao Top-3** do RRF, por pares $(consulta, documento)$. Carregado **sob demanda** apenas quando o checkbox `⚡ Ativar Cross-Encoder` é marcado.

---

## 🧩 Corpus Médico (Hardcoded)

| ID | Título | Tema |
|----|--------|------|
| Doc 1 | Protocolo Emergência ECG | ECG `CÓD-ECG-12D` na síndrome coronariana |
| Doc 2 | Guia de Farmacologia Cardíaca | AAS e antiagregantes no infarto agudo do miocárdio |
| Doc 3 | Diretriz de Hipertensão Arterial | Crise hipertensiva severa na UTI |
| Doc 4 | Manual de AVC Isquêmico | Trombolíticos em até 4h30 |
| Doc 5 | Protocolo de Reanimação RCR | Parada cardiorrespiratória e código azul |
| Doc 6 | Procedimentos de UTI Geral | Telemetria e diagnóstico `CÓD-ECG-12D` |

---

## 🖥️ Interface Streamlit

### Barra Lateral (Sidebar)
- **BM25:** sliders de $k_1$ e $b$.
- **RRF:** slider de $\alpha$ e exibição de $k_{rrf}$.
- **Bônus:** checkbox `Ativar Cross-Encoder`.
- Resumo dos parâmetros atuais.

### Área Principal
- Campo de texto para a consulta (padrão: `"infarto"`).

### Abas (`st.tabs`)
1. **🔤 Léxico (BM25):** ranking ordenado pelo score BM25, tokens extraídos da consulta e parágrafos do corpus com os tokens preservados.
2. **🧠 Semântico Vetorial:** ranking ordenado pela Similaridade de Cosseno.
3. **🔀 Híbrido RRF:** fórmula em LaTeX, parâmetros e ranking consolidado pelo Score RRF com posições léxica e semântica.
4. **📊 Matriz Comparativa:** colunas `Rank - Score BM25`, `Rank - Cosseno` e `Rank - Score RRF` + gráfico de barras dos três rankings.
5. **⚡ Cross-Encoder** *(exibida apenas com o checkbox ativo):* Top-3 antes (RRF) vs. depois (Cross-Encoder) com variação de posição e score profundo.

---

## 🛠️ Tecnologias e Bibliotecas

- **Linguagem:** Python 3.10+
- **Interface Web:** Streamlit
- **Algoritmo Léxico:** `rank-bm25`
- **Modelagem Semântica e Bônus:** `sentence-transformers`, `torch`
- **Manipulação de Dados:** `pandas`, `numpy`
- **Otimização:** `@st.cache_resource` para os modelos (Bi-Encoder e Cross-Encoder) e `@st.cache_data` para os embeddings do corpus.

---

## 🚀 Instalação e Execução

### 1. Clonar o repositório
```bash
git clone https://github.com/Perilocc/healthSearch.git
cd healthSearch
```

### 2. Criar e ativar o ambiente virtual (recomendado)
```bash
python -m venv venv
```

- **Windows (PowerShell):**
  ```bash
  .\venv\Scripts\Activate.ps1
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Executar a aplicação
```bash
streamlit run healthsearch_app.py
```

A aplicação abre no navegador em `http://localhost:8501`.

> **Nota:** na primeira execução, os modelos `paraphrase-multilingual-MiniLM-L12-v2` e (se ativado) `cross-encoder/ms-marco-MiniLM-L-6-v2` são baixados do Hugging Face Hub. A execução do Cross-Encoder sem GPU pode ser lenta.

---

## 📄 Licença

Projeto acadêmico sem fins comerciais.
