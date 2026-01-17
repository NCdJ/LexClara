# ==== Inicio imports

import re
import os
import random
import warnings
import chromadb # type: ignore
import gradio as gr # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
from typing import List, Pattern
from langchain_chroma import Chroma # type: ignore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_core.prompts import ( # type: ignore
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate   
)
from langchain_huggingface import ( # type: ignore
    ChatHuggingFace
    , HuggingFaceEmbeddings
    , HuggingFaceEndpoint
)
from transformers import logging as transformers_logging # type: ignore

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_colwidth', None)

transformers_logging.set_verbosity_info()

# ===== Configurações
chroma_db_dir = "lexclaraDB/ChromaDB"
model_llm_name = "mistralai/Mistral-7B-Instruct-v0.3"
embeddings_name = "BAAI/bge-m3"
hf_token = os.environ.get("HF_TOKEN") or os.getenv("HF_TOKEN")
# este print só é visível nos logs para garantir que o token não está vazio
if not hf_token:
    print("ERRO: O token HF_TOKEN não foi encontrado nas variáveis de ambiente!")

# ===== carregar embeddings e base de dados vetorial
model_kwargs = {'device': 'cpu'
                , 'trust_remote_code': True
                }

encode_kwargs = {'normalize_embeddings': True}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# ===== Inicializar cliente persistente
client = chromadb.PersistentClient(path=chroma_db_dir)

# Nome da coleção
collection_name = "LexClara_bge_m3_1024" # documentos breves

colecao = client.get_collection(name=collection_name)

# ===== Conectar à coleção com LangChain
vectordb = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=hf_embeddings
)

# ===== Configuração do LLM (modo serverless)
def criar_llm(
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float
):
    """
    Cria e retorna um LLM da Mistral com os parâmetros ajustáveis.

    """

    generator = HuggingFaceEndpoint(
        repo_id=model_llm_name
        ,task="text-generation"
        ,temperature=max(temperature, 0.01)
        ,top_k=top_k
        ,top_p=top_p
        ,max_new_tokens=max_tokens
        ,repetition_penalty=repetition_penalty
        ,huggingfacehub_api_token=hf_token
        ,do_sample=True
        ,streaming=True
        ,return_full_text=False
        ,connection_params={"proxies": None}
    )

    llm = ChatHuggingFace(llm=generator)
    
    return llm

# ===== Funções auxiliares

PADROES = [
    re.compile(r'(Decreto[-\s]?Lei)\s*(?:n\.º\s*)?(\d+(?:[A-Za-z-]*/\d{4}))', re.IGNORECASE),
    # re.compile(r'(Lei)\s*(?:n\.º\s*)?(\d+(?:[A-Za-z-]*/\d{4}))', re.IGNORECASE),
    re.compile(r'(Decreto\s+Regulamentar)\s*(?:n\.º\s*)?(\d+(?:[A-Za-z-]*/\d{4}))', re.IGNORECASE),
    # re.compile(r'(Portaria)\s*(?:n\.º\s*)?(\d+(?:[A-Za-z-]*/\d{4}))', re.IGNORECASE),
]

def extrair_por_regex(pergunta: str, patterns: List[Pattern]) -> List[str]:
    """
    Extrai do texto todos os identificadores completos (tipo + número),
    sem o 'n.º'. Exemplo de saída: ['Decreto-Lei 137/2023', 'Lei 12/2022'].
    """
    resultados = []
    for pat in patterns:
        for match in pat.finditer(pergunta):
            tipo = match.group(1).strip()
            numero = match.group(2).strip()
            resultados.append(f"{tipo} {numero}")
    return resultados

try:
    df_linguagem_clara_2 = pd.read_csv(filepath_or_buffer=os.path.join(os.getcwd(),'data','gold','linguagem-clara-2020-2024_2.csv'))
except Exception as e:
        print(f"Ocorreu o seguinte erro: {e}")

def gera_exemplo_aleatorio_do_df(df):

    tuplo_per_res=[]

    linha = df.sample(n=1).iloc[0]

    perguntas=[
            f"O que é o diploma {linha['identificacao_diploma']}?",
            f"O que vai mudar com o diploma {linha['identificacao_diploma']}?",
            f"Que vantagens traz o diploma {linha['identificacao_diploma']}?",
            f"Quando entra em vigor o diploma {linha['identificacao_diploma']}?"
        ]

    for i, perg in enumerate(perguntas):

        if i==0:
            res=linha['o_que_e']

        if i==1:
            res=linha['o_que_vai_mudar']

        if i==2:
            res=linha['que_vantagens_traz']
            
        if i==3:
            res=linha['quando_entra_em_vigor']

        tuplo_per_res.append((perg, res)) #type:ignore

    return tuplo_per_res

def gera_exemplo_aleatorio_para_gradio():

    pergunta, resposta = random.choice(gera_exemplo_aleatorio_do_df(df=df_linguagem_clara_2))
    
    return pergunta, resposta

def construir_prompt_few_shot() -> ChatPromptTemplate:


    # 1. Definição das mensagens de sistema e de utilizador
    sys_message = SystemMessagePromptTemplate.from_template(
        "És um chatbot de assistência jurídica que ajuda o utilizador leigo a ter contacto com as leis portuguesas.\
        Responde à pergunta com base no contexto legislativo existente na base de dados. \
        Escreve frases completas com escrita e pontuação corretas. \
        Se o contexto não contiver informação suficiente, responde exatamente:  \
        Não há informação relevante nos diplomas selecionados. \
        Não inventes respostas que não estejam no contexto. \
        Os exemplos de diálogo apresentados servem apenas para demonstrar o formato e o estilo da resposta esperada, não devem ser repetidos. \
        ")


    exemplos = gera_exemplo_aleatorio_do_df(df=df_linguagem_clara_2)

    mensagens_exemplo = []

    for entrada, resposta in exemplos:
        mensagens_exemplo.append(HumanMessagePromptTemplate.from_template(entrada))
        mensagens_exemplo.append(AIMessagePromptTemplate.from_template(resposta))

    input_prompt = HumanMessagePromptTemplate.from_template(
        "Contexto legislativo relevante:\n{context}\n\n \
        Pergunta:\n{input}\n\n \
        Resposta")

    return ChatPromptTemplate.from_messages(
        [sys_message] + mensagens_exemplo + [input_prompt]
    )

def responder_pelo_gradio_com_LLM(
    pergunta: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
):

    # Extrai os identificadores de diploma da pergunta, para serem utilizado como critério de filtragem de documentos

    ids = extrair_por_regex(pergunta, PADROES)

    if not ids:
        yield "Nenhum identificador de diploma encontrado na pergunta.", ""
        return

    retriever = vectordb.as_retriever(
        search_type="similarity"
        , search_kwargs={
            'filter': {
                'diploma': {'$in': ids}  # O mesmo filtro de metadados
            },
            'k': top_k  # O mesmo número de documentos a serem retornados
        }
    )
    
    # Obter Contexto para mostrar nos chunks
    docs = retriever.invoke(pergunta)

    if not docs:
        contexto_texto = "Não há informação relevante nos diplomas selecionados."
        chunks_para_exibir = "Nenhum segmento de texto foi encontrado na base de dados para estes IDs."
    else:
        contexto_texto = "\n\n".join([doc.page_content for doc in docs])
        chunks_para_exibir = "\n\n---\n\n".join([f"DOC {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    # Prepara o LLM com os parâmetros recebidos, e presentes no interface
    llm = criar_llm(temperature
                    , top_k
                    , top_p
                    , max_tokens
                    , repetition_penalty)
    
    # construção do prompt de auxílio ao LLM para as respostas
    prompt = construir_prompt_few_shot()


   # Chain LCEL (mais limpa para streaming)
    chain = (
        {
            "context": RunnableLambda(lambda x: contexto_texto),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Streaming da Resposta
    resposta_acumulada = ""

    for chunk in chain.stream(pergunta):
        # Limpeza de prefixos indesejados que alguns modelos inserem
        chunk_limpo = chunk.replace("AI:", "").replace("Resposta:", "")
        resposta_acumulada += chunk_limpo
        
        # O yield permite que o Gradio atualize a interface em tempo real
        # Retornamos dois valores: um para a Resposta e outro para os Segmentos
        yield resposta_acumulada, chunks_para_exibir
# ===== Interface Gradio

with gr.Blocks(title="Pergunte à Legislação com Mistral", theme=gr.themes.Default(text_size="lg")) as chatbot_LexClara: # type: ignore


    gr.Markdown("##Chat Jurídico com Mistral (Few-shot + Parametrização)")
    
    with gr.Row():
        pergunta_input = gr.Textbox(label="Pergunta"
                                    , lines = 2
                                    , placeholder="Ex: O que é o Decreto-Lei n.º 137/2023?")
        
        usar_exemplo_btn = gr.Button("Usar Exemplo Aleatório")
    
    with gr.Accordion("Parâmetros Avançados", open=False):
        temperature = gr.Slider(minimum=0
                                , maximum=1
                                , value=0.7
                                , step=0.1
                                , label="Temperatura")
        
        top_p = gr.Slider(minimum=0
                          , maximum=1
                          , value=1.0
                          , step=0.05
                          , label="Top-p")
        
        top_k = gr.Slider(minimum=1
                          , maximum=50
                          , value=5
                          , step=1
                          , label="Top-k")
        
        max_tokens = gr.Slider(minimum=100
                               , maximum=2000
                               , value=512
                               , step=100
                               , label="Número máximo de tokens gerados")
        
        repetition_penalty = gr.Slider(minimum=1.0
                                       , maximum=2.0
                                       , value=1.2
                                       , step=0.1
                                       , label="Penalização por repetição")
    
    with gr.Row():
        resposta_output = gr.Textbox(label="Resposta do LLM", lines=4)
        chunks_output = gr.Textbox(label="Segmentos de Texto Recuperados", lines=8)

    resposta_esperada_output = gr.Textbox(label="Resposta Esperada (para avaliação)", lines=4)

    perguntar_btn = gr.Button("Obter Resposta")

    # Funções aplicadas aos botões
    usar_exemplo_btn.click(
        gera_exemplo_aleatorio_para_gradio,
        inputs=[],
        outputs=[pergunta_input
                , resposta_esperada_output]
    )

    perguntar_btn.click(
        responder_pelo_gradio_com_LLM,
        inputs=[pergunta_input
                , temperature
                , top_p
                , top_k
                , max_tokens
                , repetition_penalty],
        outputs=[resposta_output
                , chunks_output]
    )


if __name__ == "__main__":
    chatbot_LexClara.launch()