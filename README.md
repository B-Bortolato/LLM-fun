
# 🤖 Support Analytics Chatbot

A powerful Streamlit-based chatbot that leverages OpenAI’s language models to interpret and answer analytical questions about support messages from platforms like LiveChat and Telegram. It understands natural language queries, extracts filters (e.g., time range, source, category), and returns relevant statistics or insights based on real message data.

---

## 🚀 Project Description

**Support Analytics Chatbot** is an intelligent assistant designed to help teams analyze and explore customer support messages through a conversational interface. It combines GPT-powered natural language understanding with backend analytics to answer questions like:

- "What is the total number of game issues in the last 4 months?"
- "What about freespin issues?"
- "Show me the last 5 messages."
- "Show me the last 5 messages regarding the deposit issues."

The chatbot extracts structured metadata from queries, performs dynamic filtering, and delivers accurate responses—mimicking an internal analytics tool with a natural chat experience.

---

## 🛠️ Features

- Chat interface powered by **Streamlit**
- GPT-driven intent detection and metadata extraction
- Dynamic query parsing (source, time range, category, etc.)
- Custom backend analytics pipeline
- JSON response parsing and metadata visualization

---

## 📄 Instructions

### 0) API Key Setup
Add your OpenAI API key in **api_key.py**:
```python
api_key = "your-openai-api-key"
```

### 1) Dataset
Provide the dataset file **LLM-DataScientist-Task_Data.csv** with the following columns:
- `id_user`: int
- `timestamp`: str (in format `YYYY/MM/DD`)
- `source`: one of `"livechat"`, `"telegram"`
- `message`: string

### 2) Installation

#### Using Conda:
```bash
conda create -n env python=3.9
conda activate env
conda install -c conda-forge openai faiss streamlit scikit-learn umap-learn numpy scipy pandas matplotlib
```

Alternatively, install the necessary packages via pip:
```bash
pip install openai faiss-cpu streamlit scikit-learn umap-learn numpy scipy pandas matplotlib
```

### 3) Run Classification First
Run the classification script to preprocess and classify the messages:
```bash
conda activate env
python classification.py
```

### 4) Run the Chatbot
Start the chatbot interface:
```bash
conda activate env
streamlit run main.py
```

---

## Project Structure

- **main.py** – Streamlit app GUI
- **backend.py** – Core backend logic for intent recognition and analytics
- **classification.py** – Preprocess and classify messages into categories
- **api_key.py** – OpenAI API key

---

## Requirements

- Python 3.9
- `openai`
- `faiss` / `faiss-cpu`
- `streamlit`
- `scikit-learn`
- `umap-learn`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
