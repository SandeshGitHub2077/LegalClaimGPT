# 🏛️ LegalClaimGPT

**LegalClaimGPT** is an AI-powered tool designed to analyze U.S. personal injury legal cases and estimate settlement outcomes based on case characteristics like injuries, medical bills, and more. The project combines NLP, predictive modeling, and explainable AI into a streamlined pipeline with an easy-to-use Streamlit frontend.

---

##  Why This Project?

Legal claims are often long, unstructured, and filled with legal jargon. LegalClaimGPT was built to help:

-  Extract meaningful case information from unstructured court documents
-  Summarize case details using Retrieval-Augmented Generation (RAG)
-  Predict estimated settlement amounts using machine learning
-  Offer SHAP-based explanations to increase trust in predictions

This helps legal professionals, claim adjusters, and researchers quickly assess the potential value of a case and understand why.

---

##  Key Features

-  Case scraping from CourtListener (US 9th Circuit)
-  Auto-filtering of personal injury cases
-  LLM-based case summarization (via RAG pipeline)
-  XGBoost regression model for predicting settlement values
-  SHAP explainability for transparency
-  FastAPI inference API
-  Streamlit interface for one-click predictions

---

## ⚙️ How It Works 

1. **Data Agent** scrapes and filters legal cases from the CourtListener API.
2. **Label Agent** uses LLMs to extract structured fields like injuries, bills, wages, etc.
3. **Summarizer** generates concise case summaries using a RAG pipeline.
4. **Model Training** is done using XGBoost on extracted features like injury type, age, medical bills, etc.
5. **Explainable AI**: SHAP plots help interpret the model’s decisions.
6. **Streamlit UI** allows users to enter case details and see predicted settlements with just one click.

---

##  Project Structure

```
LegalClaimGPT/
│
├── agents/                 # Data scraping + labeling with LLMs
│   ├── data_agent.py
│   └── label_cases.py
│
├── app/                    # API and frontend
│   ├── api.py              # FastAPI app
│   └── streamlit_app.py    # Streamlit app
│
├── data/
│   ├── raw/                # Raw cases.json
│   ├── processed/          # Summarized + labeled data
│   └── embeddings/         # FAISS index for RAG
│
├── llm/                    # RAG pipeline
│   ├── retriever.py
│   └── summarize.py
│
├── ml/                     # Model and prediction
│   ├── features.py
│   ├── train_model.py
│   ├── explain.py
│   └── predictor.py
│
├── utils/
│   ├── io_helpers.py
│   └── preprocessing.py
│
├── plots/                  # SHAP plots
│
├── run_all.bat            # One-click launcher (Windows)
├── requirements.txt
└── README.md
```

---

##  How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/LegalClaimGPT
cd LegalClaimGPT
```

### 2. Set up Environment

```bash
conda create -n LegalMedAi python=3.10
conda activate LegalMedAi
pip install -r requirements.txt
```

### 3. Run Full Pipeline

```bash
python agents/data_agent.py
python agents/label_cases.py
python llm/retriever.py
python llm/summarize.py
python ml/train_model.py
python ml/explain.py
```

### 4. Launch Frontend

```bash
streamlit run app/streamlit_app.py
```

Or double-click `run_all.bat` (on Windows)

---

##  Sample Case

```json
{
  "summary": "Plaintiff suffered a spinal cord injury due to a slip and fall.",
  "injuries": ["spinal cord injury"],
  "medical_bills": 42000,
  "lost_wages": 18000,
  "age": 46,
  "gender": "Female"
}
```

Output:

```bash
💰 Estimated Settlement: $175,340
```

---

##  License

MIT License
