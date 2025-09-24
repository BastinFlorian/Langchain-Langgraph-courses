# GCP-GenAI-Course-2025

GCP Gen AI Course - Utilisation de Google Generative AI avec LangChain

## Prérequis

- Python 3.8+
- Conda (Anaconda ou Miniconda)
- Compte Google Cloud avec accès à l'API Generative AI

## Installation

### 1. Créer l'environnement virtuel avec Conda

```bash
# Créer un nouvel environnement conda avec Python 3.11
conda create -n gcp-genai python=3.11

# Activer l'environnement
conda activate gcp-genai
```

### 2. Installer les dépendances

```bash
# Installer les packages requis
pip install langchain-google-genai langchain-core python-dotenv
```

### 3. Configuration

Créer un fichier `.env` à la racine du projet et ajouter votre clé API Google :

```
GOOGLE_API_KEY=your_api_key_here
```

## Utilisation

```bash
# Activer l'environnement conda
conda activate gcp-genai

# Exécuter le script
python test.py
```

## Désactivation de l'environnement

```bash
conda deactivate
```

## Suppression de l'environnement (optionnel)

```bash
conda remove -n gcp-genai --all
```

## 📚 Notebooks Google Colab

| # | Titre | Notions Clés | Lien Google Colab |
|---|-------|-------------|------------------|
| **LANGCHAIN FUNDAMENTALS** | | | |
| 1 | First LLM Call | API setup, ChatGoogleGenerativeAI, invoke(), temperature, max_tokens, model comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JnCNalb0J3kxROKjmYPlojVxILMzXER9) |
| 2 | Prompt Templates | PromptTemplate, ChatPromptTemplate, input variables, template formatting, system/human messages | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12j1Y1wtBInrXzz4mwSq40SEF931rj7sY) |
| 3 | Message Types | HumanMessage, AIMessage, SystemMessage, message roles, conversation structure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19w2vm7Keed24Mte6xOWLiu5hZc60ds69) |
| 4 | Structured Outputs | Pydantic models, BaseModel, type validation, structured data extraction, JSON parsing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17KUIgiIc04KXFt-XWM8pcE53a6HwqcLF) |
| 5 | Simple Chains | LCEL (pipe operator), chain composition, RunnablePassthrough, data flow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FuDhRp0csZfutFb4c9a7OjKayJdeN8ai) |
| 6 | Multimodal Inputs | Image processing, base64 encoding, vision models, multimodal prompts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oYa0xWELAZ2ip3cD24OFxU3ORMQ_da1L) |
| 7 | Streaming Responses | Real-time streaming, AIMessageChunk, chunk aggregation, streaming chains | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qlb2fKoc3uwrCnQgZ6mnZbsqurroatXN) |
| **LANGGRAPH FUNDAMENTALS** | | | |
| 1 | LangGraph Basics | StateGraph, TypedDict, nodes, edges, conditional routing, state persistence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xdA5_zYkXWcwze2PMIb-yJuMoWRXAS3-) |
| 2 | Tool Calling | Tool definitions, function calling, StructuredTool, tool integration, parameter validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh0wzVdkCvbHfgYTxxMK1kkmeZ9uisGg) |
| 3 | ReAct Prebuilt Agent | ReAct pattern, create_react_agent, agent executor, reasoning and acting, tool selection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZTe4El5fgTs6ExzE5qRj9nP-AQdO4-9_) |
| 4 | Langfuse Callback Handler | Observability, tracing, callback handlers, monitoring, performance tracking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gkkSirDi3E_I6ojuEKnHiLaTTaOA7S42) |

### 📋 Instructions d'utilisation des Notebooks

1. **Cliquez sur le badge "Open In Colab"** pour ouvrir directement le notebook dans Google Colab
2. **Exécutez la première cellule** pour installer les dépendances requises
3. **Configurez votre clé API Google** en créant un fichier `.env` ou en utilisant les secrets Colab
4. **Suivez les exercices** étape par étape dans chaque notebook

### 🔑 Prérequis pour les Notebooks

- Compte Google pour accéder à Colab
- Clé API Google Gemini (gratuite sur [Google AI Studio](https://makersuite.google.com/app/apikey))
- Connaissances de base en Python

## 🖥️ Exécution locale des scripts Python

```bash
# Limite l'output pour éviter trop de texte
python notebooks/01_langchain_fundamentals/01_first_llm_call.py
python notebooks/01_langchain_fundamentals/02_prompt_templates.py
python notebooks/01_langchain_fundamentals/03_message_types.py
python notebooks/01_langchain_fundamentals/04_structured_outputs.py
python notebooks/01_langchain_fundamentals/05_simple_chains.py
python notebooks/01_langchain_fundamentals/06_multimodal_inputs.py
python notebooks/01_langchain_fundamentals/07_streaming_responses.py


python notebooks/02_langgraph/01_langgraph_basics.py
python notebooks/02_langgraph/02_tool_calling.py
python notebooks/02_langgraph/03_react_prebuilt_agent.py
python notebooks/02_langgraph/04_langfuse_callback_handler.py
```
