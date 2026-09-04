from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import streamlit as st
import pandas as pd
import unicodedata
import re

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 HealthSearch")
st.subheader("Motor de Busca Híbrido para Protocolos Médicos")

# ============================================================
# CORPUS MÉDICO
# ============================================================

documentos = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": (
            "Pacientes com dor precordial aguda e suspeita de síndrome "
            "coronariana devem realizar eletrocardiograma CÓD-ECG-12D "
            "em até 10 minutos. O exame é fundamental para identificar "
            "alterações compatíveis com isquemia ou lesão miocárdica e "
            "deve ser realizado preferencialmente ainda na unidade de "
            "emergência. "
            
            "A avaliação inicial deve considerar a intensidade da dor, "
            "a duração dos sintomas, fatores de risco cardiovasculares "
            "e alterações nos sinais vitais. O paciente deve permanecer "
            "em monitorização cardíaca enquanto aguarda a interpretação "
            "do eletrocardiograma. Em casos de alterações sugestivas de "
            "síndrome coronariana aguda, a equipe médica deve iniciar "
            "rapidamente o protocolo de atendimento. "
            
            "O monitoramento contínuo permite identificar arritmias e "
            "outras alterações cardíacas durante a avaliação. A equipe "
            "também deve registrar o horário de início dos sintomas, "
            "os resultados do exame e as intervenções realizadas."
        )
    },

    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes "
            "plaquetários reduz a mortalidade no infarto agudo do miocárdio. "
            "A administração desses medicamentos deve considerar o quadro "
            "clínico do paciente, possíveis contraindicações e o risco "
            "de sangramento. "
            
            "O infarto agudo do miocárdio ocorre quando há redução ou "
            "interrupção do fluxo sanguíneo para uma região do músculo "
            "cardíaco. A identificação rápida dos sintomas e o início "
            "do tratamento são fundamentais para reduzir a extensão da "
            "lesão cardíaca. Dor no peito, desconforto precordial, falta "
            "de ar, sudorese e náuseas podem estar presentes. "
            
            "Além dos antiagregantes, a terapia farmacológica pode incluir "
            "outros medicamentos conforme a avaliação médica. O paciente "
            "deve permanecer sob acompanhamento da equipe de emergência "
            "e monitorização dos sinais vitais. A resposta ao tratamento "
            "deve ser avaliada continuamente durante a permanência "
            "hospitalar."
        )
    },

    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": (
            "A crise hipertensiva severa requer administração de "
            "anti-hipertensivos venosos e monitoramento contínuo "
            "da pressão arterial na UTI. A pressão arterial deve ser "
            "verificada regularmente para acompanhar a resposta ao "
            "tratamento e identificar alterações que possam representar "
            "risco ao paciente. "
            
            "Pacientes com hipertensão arterial grave podem apresentar "
            "cefaleia intensa, alterações visuais, dor torácica, falta "
            "de ar e alterações neurológicas. A avaliação clínica deve "
            "buscar sinais de lesão de órgãos-alvo, incluindo alterações "
            "cardíacas, renais e neurológicas. "
            
            "O tratamento da crise hipertensiva deve ser individualizado "
            "de acordo com a gravidade do quadro e as condições clínicas "
            "do paciente. A redução da pressão arterial deve ocorrer de "
            "forma controlada, evitando quedas excessivamente rápidas. "
            "Durante o tratamento, recomenda-se monitorização contínua "
            "e acompanhamento dos parâmetros cardiovasculares."
        )
    },

    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": (
            "O acidente vascular cerebral isquêmico agudo deve ser "
            "tratado com trombolíticos venosos em até quatro horas "
            "e meia do início dos sintomas. O reconhecimento rápido "
            "dos sinais neurológicos é essencial para determinar a "
            "possibilidade de tratamento e reduzir o risco de sequelas. "
            
            "Entre os principais sinais estão fraqueza ou dormência "
            "em um lado do corpo, dificuldade para falar, alteração "
            "da visão, perda de equilíbrio e confusão mental. A equipe "
            "de emergência deve registrar o horário em que o paciente "
            "foi visto pela última vez sem os sintomas. "
            
            "A avaliação por imagem é utilizada para auxiliar na "
            "identificação do tipo de acidente vascular cerebral e "
            "descartar situações que possam contraindicar determinados "
            "tratamentos. O acompanhamento da pressão arterial, da "
            "oxigenação e do estado neurológico deve ocorrer durante "
            "todo o atendimento."
        )
    },

    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação "
            "precoce no código azul. O reconhecimento imediato da "
            "parada e o acionamento da equipe de emergência são "
            "fundamentais para iniciar rapidamente as manobras de "
            "reanimação. "
            
            "As compressões torácicas devem ser realizadas com técnica "
            "adequada, permitindo retorno completo do tórax entre as "
            "compressões. A equipe deve avaliar o ritmo cardíaco e "
            "utilizar desfibrilação quando indicada. A ventilação deve "
            "ser realizada conforme o protocolo de atendimento e as "
            "condições do paciente. "
            
            "Durante a reanimação, é necessário manter comunicação "
            "entre os profissionais e registrar os horários das "
            "intervenções. Após o retorno da circulação espontânea, "
            "o paciente deve permanecer sob monitorização cardíaca "
            "e avaliação contínua dos sinais vitais."
        )
    },

    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca contínua "
            "por telemetria. O acompanhamento permite identificar "
            "alterações no ritmo cardíaco e auxiliar a equipe médica "
            "durante a investigação do quadro clínico. "
            
            "Pacientes internados em unidade de terapia intensiva devem "
            "ser acompanhados continuamente quanto à frequência cardíaca, "
            "pressão arterial, frequência respiratória e saturação de "
            "oxigênio. A monitorização contínua é especialmente importante "
            "em pacientes com instabilidade cardiovascular ou histórico "
            "de alterações no ritmo cardíaco. "
            
            "Em casos de arritmias complexas, o eletrocardiograma pode "
            "ser utilizado em conjunto com a telemetria para auxiliar "
            "na identificação do ritmo cardíaco. O protocolo CÓD-ECG-12D "
            "deve ser registrado no prontuário quando utilizado, "
            "permitindo relacionar os resultados do exame com a evolução "
            "clínica do paciente."
        )
    }
]

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

stopwords = {
    "a", "o", "as", "os",
    "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos",
    "em", "na", "no", "nas", "nos",
    "por", "para",
    "e", "ou",
    "com", "sem",
    "que", "se",
    "ao", "aos",
    "é",
    "sao"
}


def normalizar(texto):
    """
    Converte o texto para minúsculo
    e remove acentos.
    """

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = (
        texto
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return texto


def preprocessar(texto):
    """
    Normaliza o texto e transforma em tokens.
    """

    texto = normalizar(texto)

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+",
        texto
    )

    tokens = [
        token
        for token in tokens
        if token not in stopwords
    ]

    return tokens


# Gerando os tokens dos documentos
for documento in documentos:

    documento["tokens"] = preprocessar(
        documento["conteudo"]
    )

# ============================================================
# MODELO SEMÂNTICO
# ============================================================

@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo de embeddings.

    O cache impede que o modelo seja
    carregado novamente a cada interação.
    """

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def carregar_cross_encoder():
    """
    Carrega o modelo Cross-Encoder utilizado
    para reranking dos documentos.
    """

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

modelo = carregar_modelo()

@st.cache_data
def gerar_embeddings_documentos(textos):
    """
    Gera os embeddings dos documentos.
    """

    return modelo.encode(
        textos,
        convert_to_tensor=True
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Parâmetros da Busca")

st.sidebar.subheader("🔤 BM25")

k1 = st.sidebar.slider(
    "k₁ — Saturação da frequência",
    min_value=0.0,
    max_value=3.0,
    value=1.2,
    step=0.1
)

b = st.sidebar.slider(
    "b — Normalização pelo tamanho",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

st.sidebar.subheader("🔀 RRF")

alpha = st.sidebar.slider(
    "α — Peso do BM25",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

k_rrf = 60

st.sidebar.subheader("⭐ Bônus")

usar_cross_encoder = st.sidebar.checkbox(
    "Ativar Cross-Encoder",
    value=False
)

st.sidebar.markdown("---")
st.sidebar.write(f"**k₁ atual:** {k1}")
st.sidebar.write(f"**b atual:** {b}")
st.sidebar.write(f"**α atual:** {alpha}")
st.sidebar.write(f"**Peso Semântico:** {1 - alpha}")
st.sidebar.write(f"**k_rrf:** {k_rrf}")

# ============================================================
# CONSULTA
# ============================================================

st.header("🔎 Consulta")

consulta = st.text_input(
    "Digite sua consulta:",
    "infarto"
)

# ============================================================
# BUSCA BM25
# ============================================================

corpus_tokenizado = [
    documento["tokens"]
    for documento in documentos
]

bm25 = BM25Okapi(
    corpus_tokenizado,
    k1=k1,
    b=b
)

tokens_consulta = preprocessar(
    consulta
)

scores_bm25 = bm25.get_scores(
    tokens_consulta
)

resultados_bm25 = []

for documento, score in zip(
    documentos,
    scores_bm25
):

    resultados_bm25.append({
        "ID": documento["id"],
        "Título": documento["titulo"],
        "Score BM25": round(
            float(score),
            4
        )
    })


df_bm25 = (
    pd.DataFrame(resultados_bm25)
    .sort_values(
        by="Score BM25",
        ascending=False
    )
    .reset_index(drop=True)
)


df_bm25.insert(
    0,
    "Rank",
    range(
        1,
        len(df_bm25) + 1
    )
)

# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

textos_documentos = [
    documento["conteudo"]
    for documento in documentos
]

embeddings_documentos = gerar_embeddings_documentos(
    tuple(textos_documentos)
)

embedding_consulta = modelo.encode(
    consulta,
    convert_to_tensor=True
)

scores_semanticos = util.cos_sim(
    embedding_consulta,
    embeddings_documentos
)[0]

resultados_semanticos = []

for documento, score in zip(
    documentos,
    scores_semanticos
):

    resultados_semanticos.append({
        "ID": documento["id"],
        "Título": documento["titulo"],
        "Score Semântico": round(
            float(score),
            4
        )
    })

df_semantico = (
    pd.DataFrame(resultados_semanticos)
    .sort_values(
        by="Score Semântico",
        ascending=False
    )
    .reset_index(drop=True)
)

df_semantico.insert(
    0,
    "Rank",
    range(
        1,
        len(df_semantico) + 1
    )
)

# ============================================================
# FUSÃO RRF
# ============================================================

# Dicionário com o rank BM25
ranks_bm25 = {
    linha["ID"]: linha["Rank"]
    for _, linha in df_bm25.iterrows()
}

# Dicionário com o rank semântico
ranks_semantico = {
    linha["ID"]: linha["Rank"]
    for _, linha in df_semantico.iterrows()
}

resultados_rrf = []

for documento in documentos:
    doc_id = documento["id"]
    rank_bm25 = ranks_bm25[doc_id]
    rank_semantico = ranks_semantico[doc_id]

    componente_bm25 = (
        alpha /
        (k_rrf + rank_bm25)
    )

    componente_semantico = (
        (1 - alpha) /
        (k_rrf + rank_semantico)
    )
    
    score_rrf = (
        componente_bm25 +
        componente_semantico
    )

    resultados_rrf.append({
        "ID": doc_id,
        "Título": documento["titulo"],
        "Rank BM25": rank_bm25,
        "Rank Semântico": rank_semantico,
        "Componente BM25": round(
            componente_bm25,
            6
        ),
        "Componente Semântico": round(
            componente_semantico,
            6
        ),
        "Score RRF": round(
            score_rrf,
            6
        )
    })


df_rrf = (
    pd.DataFrame(resultados_rrf)
    .sort_values(
        by="Score RRF",
        ascending=False
    )
    .reset_index(drop=True)
)

df_rrf.insert(0, "Rank RRF", range(1, len(df_rrf) + 1))

# ============================================================
# CROSS-ENCODER — BÔNUS
# ============================================================

df_cross_encoder = None

if usar_cross_encoder:
    cross_encoder = carregar_cross_encoder()

    top_3 = df_rrf.head(3)

    pares = []

    for _, linha in top_3.iterrows():

        documento = next(
            doc for doc in documentos
            if doc["id"] == linha["ID"]
        )

        pares.append([
            consulta,
            documento["conteudo"]
        ])

    scores_cross_encoder = cross_encoder.predict(
        pares
    )

    resultados_cross_encoder = []

    for (_, linha), score in zip(
        top_3.iterrows(),
        scores_cross_encoder
    ):

        resultados_cross_encoder.append({
            "ID": linha["ID"],
            "Título": linha["Título"],
            "Rank RRF": linha["Rank RRF"],
            "Score Cross-Encoder": round(
                float(score),
                4
            )
        })

    df_cross_encoder = (
        pd.DataFrame(resultados_cross_encoder)
        .sort_values(
            by="Score Cross-Encoder",
            ascending=False
        )
        .reset_index(drop=True)
    )

    df_cross_encoder.insert(0, "Rank Final", range(1, len(df_cross_encoder) + 1))

# ============================================================
# MATRIZ COMPARATIVA
# ============================================================

mapa_scores_bm25 = {
    linha["ID"]: linha["Score BM25"]
    for _, linha in df_bm25.iterrows()
}

mapa_scores_semantico = {
    linha["ID"]: linha["Score Semântico"]
    for _, linha in df_semantico.iterrows()
}

mapa_rank_rrf = {
    linha["ID"]: (linha["Rank RRF"], linha["Score RRF"])
    for _, linha in df_rrf.iterrows()
}

df_comparativa = df_rrf[
    [
        "ID",
        "Título"
    ]
].copy()

df_comparativa["Rank - Score BM25"] = (
    df_comparativa["ID"].map(
        lambda doc_id: (
            f"{df_bm25[df_bm25['ID'] == doc_id]['Rank'].iloc[0]} - "
            f"{mapa_scores_bm25[doc_id]}"
        )
    )
)

df_comparativa["Rank Semântico - Cosseno"] = (
    df_comparativa["ID"].map(
        lambda doc_id: (
            f"{df_semantico[df_semantico['ID'] == doc_id]['Rank'].iloc[0]} - "
            f"{mapa_scores_semantico[doc_id]}"
        )
    )
)

df_comparativa["Rank - Score RRF"] = (
    df_comparativa["ID"].map(
        lambda doc_id: (
            f"{mapa_rank_rrf[doc_id][0]} - "
            f"{mapa_rank_rrf[doc_id][1]}"
        )
    )
)

# ============================================================
# INTERFACE — ABAS
# ============================================================

nomes_abas = [
    "🔤 Léxico",
    "🧠 Semântico",
    "🔀 Híbrido RRF",
    "📊 Matriz Comparativa",
]

if usar_cross_encoder:
    nomes_abas.append("⚡ Cross-Encoder")

abas = st.tabs(nomes_abas)

tab_lexico = abas[0]
tab_semantico = abas[1]
tab_rrf = abas[2]
tab_comparativa = abas[3]

if usar_cross_encoder:
    tab_cross = abas[4]

# ============================================================
# ABA 1 — LÉXICO
# ============================================================

with tab_lexico:
    st.header("🔤 Busca Léxica — BM25")

    st.write(
        "A busca léxica utiliza correspondência entre "
        "os termos da consulta e os termos presentes "
        "nos documentos."
    )

    st.info(
        f"Consulta processada: {tokens_consulta}"
    )

    st.subheader("Ranking BM25")

    st.dataframe(
        df_bm25,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("⚙️ Parâmetros utilizados")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "k₁",
            f"{k1:.2f}"
        )

    with col2:
        st.metric(
            "b",
            f"{b:.2f}"
        )

    st.markdown("---")

    st.subheader("📚 Corpus Médico")

    for documento in documentos:
        with st.expander(
            f'{documento["id"]} — {documento["titulo"]}'
        ):

            st.write(
                documento["conteudo"]
            )

            st.write(
                "**Tokens utilizados pelo BM25:**"
            )

            st.code(
                " ".join(documento["tokens"])
            )


# ============================================================
# ABA 2 — SEMÂNTICO
# ============================================================

with tab_semantico:
    st.header("🧠 Busca Semântica Vetorial")

    st.write(
        "A busca semântica representa a consulta e os "
        "documentos como vetores e calcula a similaridade "
        "de cosseno entre eles."
    )

    st.subheader("Ranking Semântico")

    st.dataframe(
        df_semantico,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📐 Método utilizado")

    st.write(
        "Modelo: "
        "`paraphrase-multilingual-MiniLM-L12-v2`"
    )

    st.write(
        "Métrica: **Similaridade de Cosseno**"
    )


# ============================================================
# ABA 3 — RRF
# ============================================================

with tab_rrf:
    st.header("🔀 Busca Híbrida — RRF")
    st.write(
        "O Reciprocal Rank Fusion combina os rankings "
        "BM25 e Semântico utilizando suas posições relativas."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "α — Peso BM25",
            f"{alpha:.2f}"
        )

    with col2:
        st.metric(
            "Peso Semântico",
            f"{1 - alpha:.2f}"
        )

    with col3:
        st.metric(
            "k_rrf",
            k_rrf
        )

    st.markdown("---")
    st.subheader("📐 Fórmula utilizada")

    st.latex(
        r"""
        Score_{RRF}(D) =
        \alpha
        \left[
        \frac{1}
        {k_{rrf}+Rank_{BM25}(D)}
        \right]
        +
        (1-\alpha)
        \left[
        \frac{1}
        {k_{rrf}+Rank_{Semantico}(D)}
        \right]
        """
    )

    st.subheader("Ranking Híbrido")
    st.dataframe(
        df_rrf,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABA 4 — MATRIZ COMPARATIVA
# ============================================================

with tab_comparativa:
    st.header("📊 Matriz Comparativa")
    st.write(
        "Comparação das posições obtidas pelos três "
        "métodos de recuperação."
    )

    st.dataframe(
        df_comparativa,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("📈 Comparação dos Rankings")

    grafico = df_rrf[
        [
            "ID",
            "Rank BM25",
            "Rank Semântico",
            "Rank RRF"
        ]
    ].set_index("ID")

    st.bar_chart(
        grafico,
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🔎 Interpretação")
    st.write(
        "Quanto menor o valor do rank, melhor a posição "
        "do documento no respectivo método."
    )
    st.write(
        "O RRF busca combinar as evidências dos dois "
        "métodos para produzir um ranking híbrido."
    )

if usar_cross_encoder:
    with tab_cross:
        st.header("⚡ Cross-Encoder Re-Ranking")

        st.write(
            "O Cross-Encoder analisa a atenção cruzada par a par "
            "entre a consulta e cada documento do Top-3 obtido "
            "pela fusão RRF."
        )

        if df_cross_encoder is not None:
            st.subheader("🔽 Ranking RRF (antes)")
            st.dataframe(
                df_rrf.head(3),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("🔼 Ranking Cross-Encoder (depois)")
            st.dataframe(
                df_cross_encoder,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("🎯 Mudança de Posição")

            mapa_titulo = {
                documento["id"]: documento["titulo"]
                for documento in documentos
            }

            comparativo_ce = []

            for _, linha in df_cross_encoder.iterrows():
                doc_id = linha["ID"]

                rank_rrf = df_rrf.loc[
                    df_rrf["ID"] == doc_id,
                    "Rank RRF"
                ].iloc[0]

                comparativo_ce.append({
                    "ID": doc_id,
                    "Título": mapa_titulo[doc_id],
                    "Rank RRF (antes)": rank_rrf,
                    "Rank Final (depois)": linha["Rank Final"],
                    "Variação": linha["Rank Final"] - rank_rrf,
                    "Score Cross-Encoder": linha["Score Cross-Encoder"]
                })

            df_comparativo_ce = pd.DataFrame(comparativo_ce)

            st.dataframe(
                df_comparativo_ce,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.info(
            "Ative o checkbox 'Ativar Cross-Encoder' para carregar "
            "o modelo e visualizar a reordenação do Top-3."
        )