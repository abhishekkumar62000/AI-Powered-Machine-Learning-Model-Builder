<img width="1024" height="1024" alt="AI" src="https://github.com/user-attachments/assets/c8e6e2d8-4e8c-42be-8cc1-e61c3a5856e9" />

<img width="1916" height="1080" alt="page main" src="https://github.com/user-attachments/assets/8b89a470-ca1e-4a36-816c-b9a36d763045" />

OUR APP:-- https://machine-learning-model-builder.streamlit.app/

https://github.com/user-attachments/assets/c1c1c1fe-2cd7-415b-94ff-2a301a2ff57b



---

# 🤖 AI-Powered Machine Learning Model Builder

***"Build, Train, and Evaluate ML Models with AI Assistance"***
🔧 Developed & Designed by: **TechSeva Solutions**
Also Known As: **📊 Data Visualization Saathi AI Agent**

---

## 📌 Introduction

In today’s world, building machine learning models typically requires coding proficiency, domain knowledge, and familiarity with a wide range of tools. However, **AI-Powered Machine Learning Model Builder** revolutionizes this process by providing a seamless, intelligent, and interactive experience that democratizes ML development.

This project is not just a typical ML dashboard—it's an **end-to-end AI agent-powered ecosystem** designed to simplify the entire machine learning lifecycle. Whether you're a beginner, data analyst, researcher, or seasoned data scientist, this application enables you to perform everything from data upload to model deployment, all from an intuitive interface powered by **LLM intelligence, interactive visualizations, and automation**.

---

## 🎯 Purpose & Vision

The core goal of this project is to:

* **Make machine learning accessible** to non-programmers.
* **Accelerate the prototyping workflow** for data scientists and ML engineers.
* Provide an **AI-assistive ML modeling environment** where you can write, interpret, and explain code automatically.
* Facilitate **interactive visual storytelling**, model explainability, and automated insights for better decision-making.

This tool isn’t just a builder—it's your intelligent **ML co-pilot**, merging human-centered design with state-of-the-art ML capabilities.

---

## 🏗️ Architectural Overview

Built on **Streamlit**, the application uses a modular, tab-based workflow to guide users across each critical stage of the ML lifecycle:

### ✅ Technologies Used:

* **Streamlit** – Interactive UI framework
* **Together.ai API** – LLM-based code assistance & explanations
* **E2B Sandbox** – Safe backend for code execution
* **Scikit-learn, XGBoost, pandas, seaborn, matplotlib, SHAP, LIME** – ML & visualization backend
* **Altair/Plotly/Matplotlib** – Customizable visualizations

---

## 🧩 Feature Breakdown (Deep Dive)

### 🔸 1. Sidebar & Branding

* **Animated logo** and vibrant dark-themed UI for branding and UX.
* **Model selection dropdown** that allows you to choose foundation models like Meta-LLaMA, DeepSeek, Qwen, etc., which guide code generation and explanation.

---

### 🔸 2. Data & Features Tab

**Functionality:**

* Upload datasets in `.csv` format and preview them.
* AI-assisted data profiling: infer data types, identify categorical vs. numerical features.
* **Convert column types** manually or using LLM suggestions.
* **Feature Selection:**

  * Manual or using algorithms like `SelectKBest`, `RFE`.
* Hyperparameter selection pane for model tuning.

**User Benefit:**

> Clean and structured data input ensures downstream modeling is efficient and high-performing.

---

### 🔸 3. Model Builder & Playground

**Functionality:**

* An **AI code generation playground** where users can input prompts like "build a classification model using Random Forest" and receive usable Python code.
* LLM explains code logic in plain English.
* You can edit, execute, and download the code within the app.

**Tech Empowerment:**

> Empowers users to learn how ML works while creating usable pipelines.

---

### 🔸 4. Data Insights

**Functionality:**

* **Auto EDA** with summary statistics, value counts, missing data visualizations.
* **Smart Recommendations** (e.g., "Consider imputing missing values using median").
* **Outlier & anomaly detection** using z-score, IQR, and isolation forest methods.
* **Correlations Matrix** and mutual information insights.

**Export:**

* Download EDA reports in `.pdf` or `.html`.

---

### 🔸 5. Visual Analytics

**Functionality:**

* **Drag-and-drop chart builder** with support for:

  * Bar, Pie, Line, Box, Violin, Scatter, Heatmap, Pairplot
* Filtering tools, color themes, data grouping.
* One-click export for plots or chart data.

**Use Case:**

> For storytelling, dashboarding, and communicating insights visually.

---

### 🔸 6. Model Comparison

**Functionality:**

* Choose and train multiple models (Logistic Regression, Random Forest, SVM, etc.)
* View performance metrics like accuracy, ROC AUC, precision, recall, F1.
* **Cross-validation & leaderboard** to compare models objectively.
* Explainable leaderboard—AI tells you why a model performed better.

---

### 🔸 7. Explainable AI (XAI)

**Functionality:**

* **SHAP & LIME** explanations show how each feature contributes to predictions.
* Per-sample explainability with waterfall and force plots.
* Permutation feature importance for model transparency.
* Export explainability reports.

**Why It Matters:**

> Builds trust in model predictions and supports fairness, interpretability, and auditability.

---

### 🔸 8. Data Cleaning Wizard

**Functionality:**

* A step-by-step wizard that walks through:

  * Handling missing values
  * Encoding categorical variables
  * Scaling and transforming data
  * Detecting/removing outliers

**End Product:**

* Fully cleaned dataset ready for modeling.
* Downloadable as `.csv`.

---

### 🔸 9. Prediction Playground

**Functionality:**

* Input your own custom data points via UI forms.
* Instantly predict outcomes using the best-trained model.
* View prediction confidence and probability distribution.

---

### 🔸 10. Model Deployment

**Functionality:**

* One-click model export (`.pkl`, `.joblib`).
* Auto-generate a shareable prediction app (Streamlit Cloud).
* Code templates for deployment via:

  * Hugging Face Spaces
  * Streamlit Sharing
  * FastAPI + Docker

---

## 🧠 AI-Powered Intelligence

* **LLM Code Copilot**: Write code with natural language.
* **Code Explanation Mode**: Learn how every line works.
* **Auto Hyperparameter Suggestion**: LLM suggests optimal parameters.
* **AutoML Builder**: Create full ML pipelines using one-click generation.

---

## 📊 Advanced Features Summary

| Feature                      | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| AutoML Pipeline              | Auto-generate end-to-end code for a classification/regression task   |
| Feature Selection Algorithms | Includes `SelectKBest`, `RFE`, `PCA` (optional)                      |
| Hyperparameter Tuning        | Sliders and selectors for advanced tuning                            |
| Exportable Artifacts         | EDA report, explanation reports, cleaned datasets, model leaderboard |

---

## 🔄 Session Management

* Tracks:

  * Uploaded datasets
  * Chat/code history
  * Model parameters
  * Current selections
  * Saved workflows
* Ensures a **continuity-focused user experience** even across sessions.

---

## 🔚 Conclusion

**AI-Powered Machine Learning Model Builder** is not just a project—it's an entire **AI-enhanced ML ecosystem**. It represents the future of how we interact with data, build models, and deploy solutions—**all in one place**.

Whether you're a:

* 🚀 Data science newbie looking to learn
* 👩‍💻 Professional building dashboards for stakeholders
* 🧠 Researcher testing model hypotheses
  —this tool puts **ML superpowers** in your hands.

---

---

## 🧠 App Workflow Architecture (LangGraph Tree)

The app is organized using a **LangGraph-inspired workflow tree**, where each node represents a functional stage or an intelligent agent in the ML lifecycle. This structure allows branching, reuse, memory, and reactivity in a modular AI agent pipeline.

```
📦 Root: AI-Powered Machine Learning Model Builder (Saathi AI Agent)
│
├── 📁 Session Manager
│   ├── Load/Save Chat & State
│   ├── Store Cleaned Data
│   ├── Track Feature Selection
│   └── Manage Hyperparameter Config
│
├── 🗂️ Data Ingestion Node
│   ├── Upload CSV
│   ├── Preview + Schema Detection
│   ├── Data Type Conversion (AI-assisted)
│   └── Feature Summary Generation
│
├── 📊 Data Insights Node
│   ├── Automated EDA Report
│   ├── Missing Value Analysis
│   ├── Anomaly & Outlier Detection
│   └── Smart Feature Correlations
│
├── 🧹 Data Cleaning Wizard
│   ├── Step-by-Step Guidance
│   ├── Handle Missing/Invalid Data
│   ├── Encoding & Normalization
│   └── Download Cleaned Dataset
│
├── 🧠 Feature Selection Agent
│   ├── Manual Column Selection
│   ├── Algorithmic Methods
│   │   ├── SelectKBest
│   │   ├── RFE
│   │   └── PCA (optional)
│   └── Save/Load Feature Sets
│
├── ⚙️ Hyperparameter Tuner
│   ├── Manual Sliders
│   ├── AI-Prompted Recommendations
│   └── Save/Apply Presets
│
├── 🧪 Model Builder & Playground
│   ├── AI Code Generator (Together API)
│   │   ├── Prompt: "Build classification model"
│   │   └── Generates editable Python code
│   ├── Code Explanation (LLM)
│   ├── Code Executor (E2B Sandbox)
│   └── Run/Debug/Download Code
│
├── 📈 Visual Analytics Engine
│   ├── Chart Types
│   │   ├── Bar, Line, Pie, Scatter, Violin, Heatmap, Box
│   ├── Dynamic Filters
│   └── Export Charts + Data
│
├── 🧪 Model Training & Comparison Node
│   ├── Train Multiple Models
│   │   ├── Logistic Regression
│   │   ├── Decision Tree
│   │   ├── Random Forest
│   │   ├── SVM
│   │   └── Custom from Code
│   ├── Leaderboard + Metrics
│   │   ├── Accuracy, Precision, Recall, F1
│   └── Cross-Validation Reports
│
├── 🔍 Explainable AI Node
│   ├── SHAP
│   ├── LIME
│   ├── Permutation Importance
│   ├── Per-Instance Force Plots
│   └── Explanation Report Export
│
├── 🔮 Prediction Playground
│   ├── User Inputs Row
│   ├── Real-Time Inference
│   ├── Confidence & Class Probabilities
│   └── Save Input/Output Logs
│
└── 🚀 Deployment Node
    ├── Export Model (.pkl/.joblib)
    ├── Generate Prediction App (UI)
    ├── Hugging Face Spaces Integration
    ├── FastAPI + Docker Template
    └── Streamlit Cloud Deployment

```

---

## 🧠 LangGraph Logic Summary

* Each node behaves like an intelligent **agent** or **tool** in a reactive tree.
* **Session Memory** enables continuity between user actions.
* **Branching logic** like `Model Builder ➝ Code ➝ Execution ➝ Visualization` is reactive.
* **LLM Integration** is used across:

  * Data profiling
  * Code generation
  * Code explanation
  * EDA and recommendation
* Secure execution via **E2B Sandboxes** allows safe and dynamic code running.

---

## 🧪 Agentic Behaviors

| Node / Agent     | Behavior                              | Tech Used               |
| ---------------- | ------------------------------------- | ----------------------- |
| Data Profiler    | Auto detects structure, types, issues | `pandas-profiling`, LLM |
| Code Generator   | Generate, explain, debug ML code      | Together API            |
| Visual Agent     | Suggests best chart types             | LLM + Altair            |
| Cleaning Wizard  | Guides data transformation            | Streamlit + Memory      |
| XAI Agent        | Explains model decisions              | SHAP, LIME              |
| Deployment Agent | Generates serving logic               | Streamlit, Hugging Face |

---

## 📌 Deployment Modes

| Platform            | Description              |
| ------------------- | ------------------------ |
| Streamlit Cloud     | For instant UI hosting   |
| Hugging Face Spaces | Share interactive demos  |
| FastAPI + Docker    | For production pipelines |

---

## ✅ Benefits of This Architecture

* ✨ **End-to-end AI lifecycle management**
* 🧠 **LLM-driven intelligence at every step**
* 🔁 **Interactive & iterative model building**
* 📊 **Robust visual + explainability support**
* ☁️ **One-click deployment options**

---

---

## ⚙️ Quick Setup Guide

### 🔧 Prerequisites

* Python 3.9+
* pip (Python package manager)
* Git (optional, for cloning)

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/abhishekkumar62000/AI-Powered-Machine-Learning-Model-Builder.git
cd ml-builder-saathi-ai
```

---

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 Includes: `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `shap`, `lime`, `together`, `e2b`, etc.

---

### 🚀 3. Run the App

```bash
streamlit run app.py
```

---

### 🌐 4. Access the App

Once running, open your browser and go to:

```
http://localhost:8501
```

---

### ✅ Optional: API Keys

To enable AI features (LLM code generation & sandbox execution):

* Add your **Together.ai API Key** to `.env`
* Add your **E2B API Key** for secure execution

Example `.env`:

```
TOGETHER_API_KEY=your_key_here
E2B_API_KEY=your_key_here
```

---




