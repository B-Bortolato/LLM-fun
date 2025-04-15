
# 🤖 QueryParser

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


# Evaluation of QueryParser

---

## 1. How did you classify feedback?

A hybrid approach is used. It consists of first embedding messages into a vector space,
then using UMAP to reduce the dimensionality of that space to 2D. After that, clustering is performed—currently, HDBSCAN is used as an example.

Clustering performance can also be evaluated visually in the dimensionally reduced space (2D), allowing careful clustering and helping avoid overfitting. Technically, clustering can be performed in the high-dimensional space, and UMAP can then be used for evaluation. After identifying clusters, a summary and a label for each are generated using GPT.

**Pros**:
- Dimensionality reduction allows visual evaluation of clustering quality.
- Classification is unsupervised—each message is assigned to a cluster.
- Dimensionality reduction is robust to overfitting.
- No assumptions about the number of clusters or categories are required.
- No manual labeling is needed—GPT generates names for each cluster.
- GPT-generated summaries for each cluster can be used in further analysis.
- Avoids overlapping categories (if clustering is well-performed).

**Cons**:
- Some information is lost during dimensionality reduction; clustering before reducing dimensions might yield better results.
- GPT may hallucinate when summarizing noisy or inconsistent clusters.
- Clustering requires manual trial and error.

**Handling New Issues**:
- New messages are embedded and projected using the same dimensionality reducer.
- In the reduced space, they are assigned to the closest cluster and inherit its label.
- If the match is poor (low confidence), messages can be flagged for review or sent to GPT for labeling.

---

## 2. How does your chatbot manage conversational context?

The system prompt sets the role of GPT, providing context for the conversation. User queries and responses are then stored and utilized to refine subsequent interactions. For each response, GPT returns a JSON object with specific parameters, which are used to trigger corresponding code. One of these parameters, named "msg_gen," stores the message generated by GPT, which is then included as part of the response to the user's query. This allows the chatbot to maintain context across multiple exchanges, ensuring coherent and relevant replies.

---

## 3. What are the main limitations?

- **Vague feedback** is difficult to classify or interpret.
- **Hallucinations** are more likely when multiple parameters need to be parsed.

---

## 4. How could the system be improved?

- Fine-tune GPT by training it on carefully curated examples and incorporating domain-adapted embeddings to improve its relevance and accuracy.

- Break down complex tasks into simpler, sequential subtasks when prompting GPT to ensure clarity and reduce ambiguity.

- Refine and optimize the system prompt to better define the context and role of GPT, leading to more accurate and coherent responses.

- Use an LLM-like model capable of reasoning and performing symbolic manipulations within defined rules to enhance problem-solving and decision-making.

---

## 5. How does the chatbot use past queries to refine current ones?

The chatbot utilizes three methods to refine current queries based on past interactions:

1) GPT is instructed to check if the current query is related to any previous queries. If a connection is found, relevant information from the conversation history is considered.

2) The current query is adjusted before being passed to the LLM by incorporating the information from the user's last response, if available.

3) The GPT output, which is in the form of a JSON object, is processed through Python code that fills in any missing information based on the last response from GPT.


---

## 6. Would your approach change if full support conversations were available?

Yes. In that case, I would first summarize the dialogue, then identify individual problems within each conversation, and extract or summarize those problems.
There may also be value in classifying entire dialogues.

---

## 7. How do you validate the correctness of message classification?

If categories are defined by summarizing the content of clusters, the primary challenge lies in clearly naming each cluster so that it’s evident which messages belong to which category, assuming the clustering process is well-executed.

Performance can also be evaluated also visually by inspecting the clusters and manually reviewing ambiguous messages. A more effective approach might be to leverage GPT to verify which label fits best, or to use other validation techniques to ensure the accuracy of the classification/clustering.
