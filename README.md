---
title: LexClara Chatbot
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
description: Assistente jurídico baseado no modelo Mistral-7B-Instruct-v0.3, utilizando RAG com uma base de dados ChromaDB de diplomas portugueses.
---

# LexClara

**LexClara** é um sistema experimental desenvolvido no contexto da minha dissertação de mestrado em Ciência de Dados. O projeto tem como objetivo simplificar a redação de diplomas legislativos &mdash; nesta fase inicial com a tipologia Decreto-Lei e Decreto Regulamentar &mdash;, tendo como objetico tornar os diplomas legais mais acessíveis a cidadãos leigos, sem comprometer o rigor e a fidelidade ao texto legal.

O sistema adota uma arquitetura de _Retrieval-Augmented Generation_ (RAG), combinando recuperação do conteúdo semântico da informação legislativa com geração de respostas através de um _Large Language Model_ (LLM) &mdash;  [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). O _pipeline_ inclui pré-processamento e segmentação de diplomas legais, recorrendo ao [LangChain](https://www.langchain.com/), geração de embeddings (com o [BGE-M3](https://huggingface.co/BAAI/bge-m3)), indexação em base de dados vetorial ([ChromaDB](https://www.trychroma.com/)), fornecendo respostas claras e contextualizadas orientadas à compreensão do conteúdo normativo pelo cidadão leigo.

Disponibilizado como _software_ de código aberto sob a licença MIT, o **LexClara** pode ser livremente utilizado, modificado e redistribuído, incluindo para fins académicos e de investigação. O projeto é disponibilizado sem garantias, não substitui aconselhamento jurídico e destina-se exclusivamente a fins educativos e experimentais, enquadrando-se na área de **LegalTech**, com foco na clareza, transparência e acessibilidade da informação legislativa.

O assistente pode ser utilizado no Hugginface Spaces (https://huggingface.co/spaces/necajesus/lexclara-chatbot).

## Tecnologias Utilizadas
- **LLM:** Mistral-7B via Hugging Face Inference API (Serverless).
- **Embeddings:** BAAI/bge-m3 (Processados localmente em CPU).
- **Vector Store:** ChromaDB.
- **Framework:** LangChain & Gradio.

## Como configurar localmente
1. Instale as dependências: `pip install -r requirements.txt`
2. Configure a variável de ambiente `HF_TOKEN`.
3. Execute: `python app.py`