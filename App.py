import matplotlib.pyplot as plt
import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Conversational AI Assistant with Contextual Memory
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}'.\nYou need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.\nIMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Add previous chat history as alternating user/assistant messages
    for q, a in st.session_state.chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    # Add current user message
    messages.append({"role": "user", "content": user_message})

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    # --- Sidebar Logo with Unique Style and Animation ---
    logo_path = os.path.join(os.path.dirname(__file__), "Logo.png")
    ai_logo_path = os.path.join(os.path.dirname(__file__), "AI.png")
    encoded_logo = None
    encoded_ai_logo = None
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_logo = base64.b64encode(image_file.read()).decode()
    if os.path.exists(ai_logo_path):
        with open(ai_logo_path, "rb") as image_file:
            encoded_ai_logo = base64.b64encode(image_file.read()).decode()

    with st.sidebar:
        if encoded_logo:
            st.markdown(
                f"""
                <style>
                @keyframes colorfulGlow {{
                    0% {{ box-shadow: 0 0 24px #ffd200, 0 0 0px #00c6ff; filter: hue-rotate(0deg); }}
                    25% {{ box-shadow: 0 0 32px #00c6ff, 0 0 12px #f7971e; filter: hue-rotate(90deg); }}
                    50% {{ box-shadow: 0 0 40px #f7971e, 0 0 24px #ffd200; filter: hue-rotate(180deg); }}
                    75% {{ box-shadow: 0 0 32px #00c6ff, 0 0 12px #ffd200; filter: hue-rotate(270deg); }}
                    100% {{ box-shadow: 0 0 24px #ffd200, 0 0 0px #00c6ff; filter: hue-rotate(360deg); }}
                }}
                .colorful-animated-logo {{
                    animation: colorfulGlow 2.5s linear infinite;
                    transition: box-shadow 0.3s, filter 0.3s;
                    border-radius: 30%;
                    box-shadow: 0 2px 12px #00c6ff;
                    border: 2px solid #ffd200;
                    background: #232526;
                    object-fit: cover;
                }}
                .sidebar-logo {{
                    text-align: center;
                    margin-bottom: 12px;
                }}
                </style>
                <div class='sidebar-logo'>
                    <img class='colorful-animated-logo' src='data:image/png;base64,{encoded_logo}' alt='Logo' style='width:150px;height:150px;'>
                    <div style='color:#00c6ff;font-size:1.1em;font-family:sans-serif;font-weight:bold;text-shadow:0 1px 6px #ffd200;margin-top:8px;'>Visualization Saathi</div>
                </div>
                <!-- Second logo below the first -->
                <div class='sidebar-AI' style='margin-top:0;'>
                    {f"<img src='data:image/png;base64,{encoded_ai_logo}' alt='AI' style='width:210px;height:220px;border-radius:30%;box-shadow:0 2px 12px #00c6ff;border:2px solid #ffd200;margin-bottom:8px;background:#232526;object-fit:cover;'>" if encoded_ai_logo else "<div style='color:#ff4b4b;'>AI.png not found</div>"}
                    <div style='color:#00c6ff;font-size:1.1em;font-family:sans-serif;font-weight:bold;text-shadow:0 1px 6px #ffd200;margin-top:8px;'></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Developer info and image below the logos
            st.markdown("<div style='text-align:center;font-size:1.1em;margin-top:10px;'>👨👨‍💻<b>Developer:</b> Abhishek💖Yadav</div>", unsafe_allow_html=True)
            developer_path = os.path.join(os.path.dirname(__file__), "pic.jpg")
            if os.path.exists(developer_path):
                st.image(developer_path, caption="Abhishek Yadav", use_container_width=True)
            else:
                st.warning("pic.jpg file not found. Please check the file path.")
        else:
            st.markdown(
                "<div style='text-align:center;font-size:2em;margin:16px 0;'>🚀</div><div style='text-align:center;color:#00c6ff;font-weight:bold;'>NewsCraft.AI</div>",
                unsafe_allow_html=True
            )
    # Main Streamlit application logic (docstring removed from UI)

    # --- Custom Dark Colorful UI/UX Styling ---
    st.markdown(
        """
        <style>
        /* App background and main container */
        .stApp {
            background: linear-gradient(135deg, #1a093e 0%, #0f2027 100%);
            color: #e0e0e0;
            font-family: 'Montserrat', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        }
        /* Animated Glowing Title */
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00f2fe, #4a00e0, #8f94fb, #f7971e, #f953c6, #43e97b, #38f9d7);
            background-size: 400% auto;
            color: #fff;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow-move 5s linear infinite, glow-shadow 2s alternate infinite;
            text-align: center;
            margin-bottom: 0.5em;
            filter: drop-shadow(0 0 16px #00f2fe88);
        }
        @keyframes glow-move {
            0% {background-position: 0% 50%;}
            100% {background-position: 100% 50%;}
        }
        @keyframes glow-shadow {
            0% {filter: drop-shadow(0 0 16px #00f2fe88);}
            100% {filter: drop-shadow(0 0 32px #f953c688);}
        }
        /* Subtitle */
        .subtitle {
            color: #43e97b;
            font-size: 1.25rem;
            text-align: center;
            margin-bottom: 1.5em;
            letter-spacing: 1px;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: #fff;
            border-radius: 18px 0 0 18px;
        }
        /* Dataframe and code blocks */
        .stDataFrame, .stTable, .stCodeBlock, .stMarkdown pre {
            background: #1a093e !important;
            color: #43e97b !important;
            border-radius: 12px;
            box-shadow: 0 2px 12px #38f9d733;
        }
        /* Chat bubbles (override for both user and AI) */
        div[style*='background:linear-gradient(90deg,#232526,#414345)'],
        div[style*='background:linear-gradient(90deg,#141e30,#243b55)'] {
            animation: fadeIn 0.7s;
            background: linear-gradient(90deg, #8f94fb 0%, #4a00e0 100%) !important;
            color: #fff !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 16px #00f2fe44;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* Download buttons */
        .stDownloadButton>button {
            background: linear-gradient(90deg, #8f94fb 0%, #43e97b 100%);
            color: #1a093e;
            border-radius: 10px;
            font-weight: 700;
            transition: background 0.2s, color 0.2s;
            box-shadow: 0 2px 12px #43e97b66;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #f953c6 0%, #00f2fe 100%);
            color: #fff;
        }
        /* Inputs */
        .stTextInput>div>div>input, .stTextArea textarea {
            background: #1a093e;
            color: #fff;
            border: 2px solid #43e97b;
            border-radius: 10px;
        }
        /* Checkbox */
        .stCheckbox>label>div:first-child {
            border: 2px solid #f953c6;
        }
        /* Subheaders */
        .stSubheader {
            color: #8f94fb;
        }
        /* Markdown links */
        .stMarkdown a {
            color: #43e97b;
            text-decoration: underline;
        }
        /* Animations for tab content */
        .block-container {
            animation: fadeIn 0.8s;
        }
        /* Animated card backgrounds for info/success/warning */
        .stAlert, .stInfo, .stSuccess, .stWarning {
            background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%) !important;
            color: #1a093e !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 16px #38f9d799;
            animation: fadeIn 0.8s;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<div class="main-title">🤖 AI-Powered Machine Learning Model Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your dataset, select features, and let AI build, train, and evaluate ML models for you!</div>', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .tagline {
            font-size: 1.25rem;
            font-weight: 600;
            background: linear-gradient(90deg, #ffd200, #00c6ff, #43e97b, #f953c6);
            background-size: 300% auto;
            color: #fff;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: tagline-glow 4s linear infinite;
            text-align: center;
            margin-bottom: 0.5em;
            letter-spacing: 1px;
            filter: drop-shadow(0 0 8px #ffd20088);
        }
        @keyframes tagline-glow {
            0% {background-position: 0% 50%;}
            100% {background-position: 100% 50%;}
        }
        .powered-by {
            text-align: center;
            font-size: 1.05rem;
            font-weight: 500;
            color: #00c6ff;
            margin-bottom: 1.2em;
            letter-spacing: 0.5px;
            text-shadow: 0 1px 8px #ffd20099, 0 0px 2px #232526;
        }
        </style>
        <div class="tagline">Build, Train, and Evaluate ML Models with AI Assistance</div>
        <div class="powered-by">🚀 Powered by <span style='color:#ffd200;font-weight:bold;'>TechSeva Solutions</span></div>
    """, unsafe_allow_html=True)

    # Load .env variables
    load_dotenv()

    # Initialize session state variables
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = os.environ.get('TOGETHER_API_KEY', '')
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = os.environ.get('E2B_API_KEY', '')
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''
    # Chat history: list of (question, answer) tuples
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # Initialize selected features and target to avoid AttributeError
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None

    with st.sidebar:
        # Only show model selection dropdown
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    # --- Main App Tabs ---
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or invalid. Please upload a valid CSV file with data.")
            st.session_state.df = None
            st.session_state.uploaded_file_name = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state.df = None
            st.session_state.uploaded_file_name = None
    df = st.session_state.df

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📁 Data & Features",
        "🤖 Model Builder & Playground",
        "📊 Data Insights",
        "📈 Visual Analytics",
        "🏆 Model Comparison",
        "🔍 Explainable AI",
        "🧹 Data Cleaning Wizard",
        "🧑‍🔬 Prediction Playground",
        "🚀 Model Deployment"
    ])
    # --- Prediction Playground Tab ---
    with tab8:
        st.header("🧑‍🔬 Prediction Playground")
        st.info("Test new data and get instant predictions using the best trained model.")
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.write("Enter values for each feature:")
            input_data = {}
            for feature in st.session_state.selected_features:
                dtype = df[feature].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
                else:
                    input_data[feature] = st.text_input(f"{feature}", value=str(df[feature].mode()[0]))
            if st.button("Predict"):
                # Use best model from AutoML if available, else train a default model
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                import numpy as np
                X = df[st.session_state.selected_features].copy()
                y = df[st.session_state.selected_target].copy()
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = X[col].astype('category').cat.codes
                if y.dtype == 'object':
                    y = y.astype('category').cat.codes
                # Train model (Random Forest)
                if y.nunique() > 10:
                    model = RandomForestRegressor(n_estimators=100)
                else:
                    model = RandomForestClassifier(n_estimators=100)
                model.fit(X, y)
                # Prepare input for prediction
                input_df = pd.DataFrame([input_data])
                for col in input_df.select_dtypes(include=['object']).columns:
                    input_df[col] = input_df[col].astype('category').cat.codes
                pred = model.predict(input_df)[0]
                st.success(f"Prediction: {pred}")
        else:
            st.info("Upload a CSV and select features/target to use the playground.")

    # --- Model Deployment Tab ---
    with tab9:
        st.header("🚀 Model Deployment & Sharing")
        st.info("Export your trained model as a file or get a shareable link for predictions.")
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            import joblib
            X = df[st.session_state.selected_features].copy()
            y = df[st.session_state.selected_target].copy()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category').cat.codes
            if y.dtype == 'object':
                y = y.astype('category').cat.codes
            # Train model (Random Forest)
            if y.nunique() > 10:
                model = RandomForestRegressor(n_estimators=100)
            else:
                model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            # Save model to file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                joblib.dump(model, tmp.name)
                with open(tmp.name, 'rb') as f:
                    st.download_button("Download Model File (.joblib)", data=f.read(), file_name="trained_model.joblib")
            st.markdown("---")
            st.subheader("Shareable Link (Demo)")
            st.info("To deploy as an API or web app, use platforms like Hugging Face Spaces, Streamlit Cloud, or FastAPI. Here is a demo link:")
            st.markdown("[Deploy on Hugging Face Spaces](https://huggingface.co/spaces)")
        else:
            st.info("Upload a CSV and select features/target to deploy your model.")
    # --- Data Cleaning Wizard Tab ---
    with tab7:
        st.header("🧹 Data Cleaning Wizard")
        st.info("Clean your data step-by-step for better model performance. Handle missing values, outliers, and encoding interactively.")
        if df is not None:
            # Initialize cleaned_df in session state if not present
            if 'cleaned_df' not in st.session_state:
                st.session_state.cleaned_df = df.copy()
            st.subheader("Step 1: Missing Value Handling")
            missing = df.isnull().sum()
            st.dataframe(missing)
            missing_cols = missing[missing > 0].index.tolist()
            if missing_cols:
                method = st.selectbox("Choose imputation method", ["Mean/Mode Imputation", "Drop Rows", "Drop Columns"])
                if st.button("Apply Missing Value Handling"):
                    cleaned_df = df.copy()
                    if method == "Mean/Mode Imputation":
                        for col in missing_cols:
                            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                            else:
                                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                        st.success("Missing values imputed.")
                    elif method == "Drop Rows":
                        cleaned_df.dropna(inplace=True)
                        st.success("Rows with missing values dropped.")
                    elif method == "Drop Columns":
                        cleaned_df.drop(columns=missing_cols, inplace=True)
                        st.success(f"Columns {missing_cols} dropped.")
                    st.dataframe(cleaned_df.head())
                    st.session_state.cleaned_df = cleaned_df
            else:
                st.success("No missing values detected.")
                st.session_state.cleaned_df = df.copy()
            st.markdown("---")
            st.subheader("Step 2: Outlier Detection & Handling")
            numeric_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
            z = (st.session_state.cleaned_df[outlier_col] - st.session_state.cleaned_df[outlier_col].mean()) / st.session_state.cleaned_df[outlier_col].std()
            outliers = st.session_state.cleaned_df[abs(z) > 3]
            st.write(f"Outliers detected: {outliers.shape[0]}")
            outlier_action = st.selectbox("Choose outlier handling method", ["None", "Remove Outliers", "Cap Values"])
            if st.button("Apply Outlier Handling"):
                cleaned_df2 = st.session_state.cleaned_df.copy()
                if outlier_action == "Remove Outliers":
                    cleaned_df2 = cleaned_df2[abs(z) <= 3]
                    st.success("Outliers removed.")
                elif outlier_action == "Cap Values":
                    cap_low = cleaned_df2[outlier_col].mean() - 3 * cleaned_df2[outlier_col].std()
                    cap_high = cleaned_df2[outlier_col].mean() + 3 * cleaned_df2[outlier_col].std()
                    cleaned_df2[outlier_col] = cleaned_df2[outlier_col].clip(lower=cap_low, upper=cap_high)
                    st.success("Outliers capped to 3 std dev.")
                st.dataframe(cleaned_df2.head())
                st.session_state.cleaned_df = cleaned_df2
            st.markdown("---")
            st.subheader("Step 3: Categorical Encoding")
            cat_cols = st.session_state.cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                encoding_method = st.selectbox("Choose encoding method", ["Label Encoding", "One-Hot Encoding"])
                if st.button("Apply Encoding"):
                    cleaned_df3 = st.session_state.cleaned_df.copy()
                    if encoding_method == "Label Encoding":
                        for col in cat_cols:
                            cleaned_df3[col] = cleaned_df3[col].astype('category').cat.codes
                        st.success("Label encoding applied.")
                    elif encoding_method == "One-Hot Encoding":
                        cleaned_df3 = pd.get_dummies(cleaned_df3, columns=cat_cols)
                        st.success("One-hot encoding applied.")
                    st.dataframe(cleaned_df3.head())
                    st.session_state.cleaned_df = cleaned_df3
            else:
                st.info("No categorical columns detected.")
            st.markdown("---")
            st.subheader("Step 4: Data Transformation")
            transform_col = st.selectbox("Select column for transformation", st.session_state.cleaned_df.columns)
            transform_method = st.selectbox("Choose transformation", ["None", "Log Transform", "Standard Scaling", "Min-Max Scaling"])
            if st.button("Apply Transformation"):
                cleaned_df4 = st.session_state.cleaned_df.copy()
                if transform_method == "Log Transform":
                    cleaned_df4[transform_col] = np.log1p(cleaned_df4[transform_col].clip(lower=0))
                    st.success("Log transform applied.")
                elif transform_method == "Standard Scaling":
                    cleaned_df4[transform_col] = (cleaned_df4[transform_col] - cleaned_df4[transform_col].mean()) / cleaned_df4[transform_col].std()
                    st.success("Standard scaling applied.")
                elif transform_method == "Min-Max Scaling":
                    cleaned_df4[transform_col] = (cleaned_df4[transform_col] - cleaned_df4[transform_col].min()) / (cleaned_df4[transform_col].max() - cleaned_df4[transform_col].min())
                    st.success("Min-max scaling applied.")
                st.dataframe(cleaned_df4.head())
                st.session_state.cleaned_df = cleaned_df4
            st.markdown("---")
            st.subheader("Step 5: Download Cleaned Data")
            st.download_button("Download Cleaned Data", data=st.session_state.cleaned_df.to_csv(index=False), file_name="cleaned_data.csv")
        else:
            st.info("Upload a CSV file to use the Data Cleaning Wizard.")
    # --- AutoML Pipeline Builder Tab ---
    st.markdown("""
        <style>
        .automl-title {
            font-size: 2.2rem;
            font-weight: bold;
            background: linear-gradient(90deg, #ffd200, #00c6ff, #43e97b, #f953c6);
            background-size: 300% auto;
            color: #fff;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: tagline-glow 4s linear infinite;
            text-align: center;
            margin-bottom: 0.5em;
            letter-spacing: 1px;
            filter: drop-shadow(0 0 8px #ffd20088);
        }
        </style>
    """, unsafe_allow_html=True)
    with st.expander("🚀 AutoML Pipeline Builder (One-Click ML)", expanded=True):
        st.markdown('<div class="automl-title">AutoML Pipeline Builder</div>', unsafe_allow_html=True)
        st.info("Automatically build, tune, and select the best ML model for your data in one click!")
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.write("### Select Task Type")
            target_type = 'classification' if df[st.session_state.selected_target].dtype == 'object' or df[st.session_state.selected_target].nunique() < 10 else 'regression'
            st.write(f"Detected task: **{target_type.capitalize()}**")
            st.write("### AutoML Models to Try")
            if target_type == 'classification':
                automl_models = ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "Gradient Boosting"]
            else:
                automl_models = ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "Gradient Boosting"]
            st.write(automl_models)
            st.write("### AutoML Settings")
            test_size = st.slider("Test Size (%)", 10, 50, 20)
            run_automl = st.button("Run AutoML Pipeline")
            if run_automl:
                from sklearn.model_selection import train_test_split, GridSearchCV
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
                from sklearn.linear_model import LogisticRegression, LinearRegression
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.svm import SVC, SVR
                import numpy as np
                X = df[st.session_state.selected_features].copy()
                y = df[st.session_state.selected_target].copy()
                # Encode categorical features if needed
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = X[col].astype('category').cat.codes
                if y.dtype == 'object':
                    y = y.astype('category').cat.codes
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                leaderboard = []
                model_dict = {}
                # For classification, set cv to min(3, smallest class size)
                cv_value = 3
                if target_type == 'classification':
                    unique, counts = np.unique(y_train, return_counts=True)
                    min_class_size = counts.min() if len(counts) > 0 else 1
                    cv_value = min(3, min_class_size)
                    if cv_value < 2:
                        st.error("Not enough samples in each class for cross-validation. Please use a larger dataset or reduce test size.")
                        return
                for model_name in automl_models:
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                        param_grid = {"C": [0.1, 1, 10]}
                    elif model_name == "Random Forest":
                        if target_type == 'classification':
                            model = RandomForestClassifier()
                            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 7, None]}
                        else:
                            model = RandomForestRegressor()
                            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 7, None]}
                    elif model_name == "Decision Tree":
                        if target_type == 'classification':
                            model = DecisionTreeClassifier()
                            param_grid = {"max_depth": [3, 7, None]}
                        else:
                            model = DecisionTreeRegressor()
                            param_grid = {"max_depth": [3, 7, None]}
                    elif model_name == "Support Vector Machine":
                        if target_type == 'classification':
                            model = SVC(probability=True)
                            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                        else:
                            model = SVR()
                            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                    elif model_name == "Gradient Boosting":
                        if target_type == 'classification':
                            model = GradientBoostingClassifier()
                            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 7, None]}
                        else:
                            model = GradientBoostingRegressor()
                            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 7, None]}
                    else:
                        continue
                    grid = GridSearchCV(model, param_grid, cv=cv_value, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    model_dict[model_name] = best_model
                    y_pred = best_model.predict(X_test)
                    metrics = {}
                    if target_type == 'classification':
                        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                        metrics['F1'] = f1_score(y_test, y_pred, average='weighted')
                        try:
                            if hasattr(best_model, "predict_proba"):
                                metrics['ROC AUC'] = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovo')
                        except Exception:
                            metrics['ROC AUC'] = 'N/A'
                    else:
                        metrics['MSE'] = mean_squared_error(y_test, y_pred)
                    leaderboard.append({'Model': model_name, **metrics, 'Best Params': grid.best_params_})
                leaderboard_df = pd.DataFrame(leaderboard)
                st.dataframe(leaderboard_df)
                # Highlight best model
                if not leaderboard_df.empty:
                    if target_type == 'classification' and 'Accuracy' in leaderboard_df.columns:
                        best_idx = leaderboard_df['Accuracy'].idxmax()
                        st.success(f"Best Model: {leaderboard_df.loc[best_idx, 'Model']} (Accuracy: {leaderboard_df.loc[best_idx, 'Accuracy']:.3f})")
                    elif target_type == 'regression' and 'MSE' in leaderboard_df.columns:
                        best_idx = leaderboard_df['MSE'].idxmin()
                        st.success(f"Best Model: {leaderboard_df.loc[best_idx, 'Model']} (MSE: {leaderboard_df.loc[best_idx, 'MSE']:.3f})")
                st.download_button("Download AutoML Leaderboard", data=leaderboard_df.to_csv(index=False), file_name="automl_leaderboard.csv")
                st.info("AutoML pipeline completed! You can now use the best model for predictions or further analysis.")
        else:
            st.info("Upload a CSV and select features/target to use AutoML.")
    # --- Model Comparison & Leaderboard Tab ---
    with tab5:
        st.header("🏆 Model Comparison & Leaderboard")
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.subheader("Select Models to Compare")
            compare_models = st.multiselect("Models", ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "Linear Regression"], default=[st.session_state.selected_model_type])
            st.info("Click 'Run Comparison' to train and compare selected models.")
            if st.button("Run Comparison"):
                import sklearn
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
                from sklearn.linear_model import LogisticRegression, LinearRegression
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.svm import SVC, SVR
                X = df[st.session_state.selected_features]
                y = df[st.session_state.selected_target]
                # Encode categorical features if needed
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = X[col].astype('category').cat.codes
                if y.dtype == 'object':
                    y = y.astype('category').cat.codes
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                results = []
                for model_name in compare_models:
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "Random Forest":
                        if y.nunique() > 10:
                            model = RandomForestRegressor(n_estimators=100)
                        else:
                            model = RandomForestClassifier(n_estimators=100)
                    elif model_name == "Decision Tree":
                        if y.nunique() > 10:
                            model = DecisionTreeRegressor()
                        else:
                            model = DecisionTreeClassifier()
                    elif model_name == "Support Vector Machine":
                        if y.nunique() > 10:
                            model = SVR()
                        else:
                            model = SVC(probability=True)
                    elif model_name == "Linear Regression":
                        model = LinearRegression()
                    else:
                        continue
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {}
                    if y.nunique() > 10:
                        metrics['MSE'] = mean_squared_error(y_test, y_pred)
                    else:
                        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                        metrics['F1'] = f1_score(y_test, y_pred, average='weighted')
                        try:
                            if hasattr(model, "predict_proba"):
                                metrics['ROC AUC'] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo')
                        except Exception:
                            metrics['ROC AUC'] = 'N/A'
                    results.append({'Model': model_name, **metrics})
                leaderboard = pd.DataFrame(results)
                st.dataframe(leaderboard)
                # Highlight best model
                if not leaderboard.empty:
                    if 'Accuracy' in leaderboard.columns:
                        best_idx = leaderboard['Accuracy'].idxmax()
                        st.success(f"Best Model: {leaderboard.loc[best_idx, 'Model']} (Accuracy: {leaderboard.loc[best_idx, 'Accuracy']:.3f})")
                    elif 'MSE' in leaderboard.columns:
                        best_idx = leaderboard['MSE'].idxmin()
                        st.success(f"Best Model: {leaderboard.loc[best_idx, 'Model']} (MSE: {leaderboard.loc[best_idx, 'MSE']:.3f})")
                st.download_button("Download Leaderboard", data=leaderboard.to_csv(index=False), file_name="leaderboard.csv")
        else:
            st.info("Upload a CSV and select features/target to compare models.")

    # --- Explainable AI (SHAP) Tab ---
    with tab6:
        st.header("🔍 Explainable AI (SHAP) & Feature Importance")
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.subheader("Train Model for Explanation")
            import shap
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            X = df[st.session_state.selected_features]
            y = df[st.session_state.selected_target]
            # Encode categorical features if needed
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category').cat.codes
            if y.dtype == 'object':
                y = y.astype('category').cat.codes
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if y.nunique() > 10:
                model = RandomForestRegressor(n_estimators=100)
            else:
                model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            st.success("Model trained. Showing SHAP feature importance plot below:")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            import matplotlib
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(plt.gcf())
            st.info("SHAP shows which features most influence model predictions. For more details, see SHAP documentation.")
        else:
            st.info("Upload a CSV and select features/target to view explanations.")

    # --- Initialize feature engineering session state ---
    if 'selected_scaler' not in st.session_state:
        st.session_state.selected_scaler = "None"
    if 'selected_encoder' not in st.session_state:
        st.session_state.selected_encoder = "None"
    # --- Initialize hyperparameters for all models ---
    if 'n_estimators' not in st.session_state:
        st.session_state.n_estimators = 100
    if 'max_depth' not in st.session_state:
        st.session_state.max_depth = 5
    if 'c_value' not in st.session_state:
        st.session_state.c_value = 1.0

    with tab1:
        if df is not None:
            st.write("### Uploaded Dataset Preview")
            st.dataframe(df.head())
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write("**Column Types:**")
            st.write(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.values}))
            # Feature and target selection
            st.subheader("Select Features and Target")
            columns = df.columns.tolist()
            target = st.selectbox("Target Column (what to predict)", columns)
            features = st.multiselect("Feature Columns (inputs)", [col for col in columns if col != target], default=[col for col in columns if col != target][:min(3, len(columns)-1)])
            st.session_state.selected_target = target
            st.session_state.selected_features = features
            # --- Enhanced Model Selection ---
            st.subheader("Choose ML Algorithm")
            # Detect target type for classification/regression
            target_type = None
            if df[target].dtype == 'object' or df[target].nunique() < 10:
                target_type = 'classification'
            else:
                target_type = 'regression'
            if target_type == 'classification':
                model_options = ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]
            else:
                model_options = ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]
            model_type = st.selectbox("Model Type", model_options)
            st.session_state.selected_model_type = model_type
            # --- Dynamic Hyperparameter Controls ---
            st.markdown("#### ⚙️ Hyperparameter Tuning")
            if model_type == "Random Forest":
                n_estimators = st.slider("n_estimators", 10, 200, st.session_state.n_estimators)
                max_depth = st.slider("max_depth", 1, 20, st.session_state.max_depth)
                st.session_state.n_estimators = n_estimators
                st.session_state.max_depth = max_depth
            elif model_type == "Decision Tree":
                max_depth = st.slider("max_depth", 1, 20, st.session_state.max_depth)
                st.session_state.max_depth = max_depth
            elif model_type == "Support Vector Machine":
                c_value = st.slider("C (Regularization)", 0.01, 10.0, st.session_state.c_value)
                st.session_state.c_value = c_value
            st.success(f"Selected Target: {target} ({target_type}), Features: {features}, Model: {model_type}")
            # --- User Guidance ---
            if target_type == 'classification' and model_type == 'Linear Regression':
                st.warning("Linear Regression is not suitable for classification tasks. Please select a classification model.")
            if target_type == 'regression' and model_type == 'Logistic Regression':
                st.warning("Logistic Regression is not suitable for regression tasks. Please select a regression model.")
            if not features:
                st.info("Please select at least one feature column.")
        else:
            st.info("Upload a CSV file to get started.")

    # --- Data Insights Tab ---
    with tab3:
        st.header("📊 Automated Data Insights & Smart Recommendations")
        if df is not None:
            st.subheader("Missing Value Analysis")
            missing = df.isnull().sum()
            st.dataframe(missing)
            if missing.sum() > 0:
                st.warning("Some columns have missing values. Consider imputing or dropping them.")
            else:
                st.success("No missing values detected.")
            st.subheader("Outlier Detection (Z-score)")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            outlier_info = {}
            for col in numeric_cols:
                z = (df[col] - df[col].mean()) / df[col].std()
                outliers = df[abs(z) > 3]
                outlier_info[col] = outliers.shape[0]
            st.write(outlier_info)
            if any(v > 0 for v in outlier_info.values()):
                st.warning("Outliers detected in some columns. Consider handling them.")
            else:
                st.success("No significant outliers detected.")
            st.subheader("Feature Correlations")
            corr = df.corr(numeric_only=True)
            st.dataframe(corr)
            st.markdown("**Highly correlated features (>|0.8|):**")
            high_corr = corr[(abs(corr) > 0.8) & (abs(corr) < 1.0)]
            if not high_corr.empty:
                st.write(high_corr)
                st.info("Consider removing or combining highly correlated features.")
            else:
                st.success("No highly correlated features detected.")
            st.subheader("Recommended Model Type")
            if df[target].dtype == 'object' or df[target].nunique() < 10:
                st.info("Recommended: Classification models (Logistic Regression, Random Forest, etc.)")
            else:
                st.info("Recommended: Regression models (Linear Regression, Random Forest, etc.)")
        else:
            st.info("Upload a CSV file to see insights.")

    # --- Visual Analytics Tab ---
    with tab4:
        st.header("📈 Interactive Visual Analytics Dashboard")
        if df is not None:
            st.subheader("Choose Chart Type")
            chart_type = st.selectbox("Chart Type", ["Bar", "Scatter", "Box", "Heatmap"])
            x_col = st.selectbox("X Axis", df.columns)
            y_col = st.selectbox("Y Axis", [col for col in df.columns if col != x_col])
            if chart_type == "Bar":
                # Only aggregate numeric columns
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    bar_df = df[[x_col, y_col]].groupby(x_col)[y_col].mean().reset_index()
                    st.bar_chart(bar_df.set_index(x_col))
                else:
                    st.warning(f"Column '{y_col}' is not numeric. Please select a numeric column for Y axis.")
            elif chart_type == "Scatter":
                st.write(plt.figure())
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col], alpha=0.7)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
            elif chart_type == "Box":
                st.write(plt.figure())
                fig, ax = plt.subplots()
                ax.boxplot(df[y_col].dropna())
                ax.set_title(f"Boxplot of {y_col}")
                st.pyplot(fig)
            elif chart_type == "Heatmap":
                import seaborn as sns
                st.write(plt.figure())
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            st.subheader("Filter Data")
            filter_col = st.selectbox("Filter Column", df.columns)
            filter_val = st.text_input("Filter Value (exact match)")
            if filter_val:
                filtered_df = df[df[filter_col].astype(str) == filter_val]
                st.dataframe(filtered_df)
            st.download_button("Download Chart Data", data=df.to_csv(index=False), file_name="chart_data.csv")
        else:
            st.info("Upload a CSV file to visualize data.")

    with tab2:
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.subheader("AI Model Builder & Code Playground")
            # Prompt for LLM to generate ML code
            # --- Build ML Prompt with Feature Engineering and Hyperparameters ---
            ml_prompt = f"""
You are a Python machine learning expert. Given a dataset at path './{st.session_state.uploaded_file_name}', write code to:
- Load the CSV
- Apply feature scaling: {st.session_state.selected_scaler}
- Apply categorical encoding: {st.session_state.selected_encoder}
- Split into train/test
- Train a {st.session_state.selected_model_type} model to predict '{st.session_state.selected_target}' using features {st.session_state.selected_features}
- Use these hyperparameters: """
            if st.session_state.selected_model_type == "Random Forest":
                ml_prompt += f"n_estimators={st.session_state.n_estimators}, max_depth={st.session_state.max_depth}\n"
            elif st.session_state.selected_model_type == "Decision Tree":
                ml_prompt += f"max_depth={st.session_state.max_depth}\n"
            elif st.session_state.selected_model_type == "Support Vector Machine":
                ml_prompt += f"C={st.session_state.c_value}\n"
            ml_prompt += f"- Print train/test accuracy or relevant metrics\n- Show feature importances if available\n- Plot confusion matrix, ROC curve, and other relevant evaluation visualizations\nIMPORTANT: Always use the dataset path variable './{st.session_state.uploaded_file_name}' in your code.\n"""
            st.write("#### AI will generate code for your selected ML task:")
            st.code(ml_prompt)
            if st.button("Generate & Run ML Code"):
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    code_results, llm_response = chat_with_llm(code_interpreter, ml_prompt, dataset_path)
                    st.session_state.chat_history.append((ml_prompt, llm_response))
                    st.write("AI Response:")
                    st.write(llm_response)
                    # Playground for code
                    code_block = match_code_blocks(llm_response)
                    if code_block:
                        st.markdown("---")
                        st.markdown("### 🛠️ Code Playground")
                        st.info("You can edit, run, reset, or download the code below. Try your own ideas!")
                        playground_key = f"playground_{len(st.session_state.chat_history)}"
                        if f"original_code_{playground_key}" not in st.session_state:
                            st.session_state[f"original_code_{playground_key}"] = code_block
                        if f"user_code_{playground_key}" not in st.session_state:
                            st.session_state[f"user_code_{playground_key}"] = code_block
                        playground_output_key = f"playground_output_{playground_key}"
                        playground_error_key = f"playground_error_{playground_key}"
                        with st.form(key=f"form_{playground_key}"):
                            st.code(st.session_state[f"user_code_{playground_key}"], language="python")
                            col1, col2 = st.columns([1,1])
                            run_clicked = col1.form_submit_button("▶️ Run Code")
                            reset_clicked = col2.form_submit_button("🔄 Reset")
                            user_code = st.text_area(
                                "Python code from AI (editable)",
                                value=st.session_state[f"user_code_{playground_key}"],
                                height=220,
                                key=f"text_area_{playground_key}"
                            )
                            if run_clicked or reset_clicked:
                                if reset_clicked:
                                    st.session_state[f"user_code_{playground_key}"] = st.session_state[f"original_code_{playground_key}"]
                                    st.session_state[playground_output_key] = None
                                    st.session_state[playground_error_key] = None
                                else:
                                    st.session_state[f"user_code_{playground_key}"] = user_code
                                if run_clicked:
                                    with st.spinner("Running your code in a secure sandbox..."):
                                        try:
                                            with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter_playground:
                                                result = code_interpret(code_interpreter_playground, st.session_state[f"user_code_{playground_key}"])
                                                st.session_state[playground_output_key] = result
                                                st.session_state[playground_error_key] = None
                                        except Exception as e:
                                            st.session_state[playground_output_key] = None
                                            st.session_state[playground_error_key] = str(e)
                        st.download_button(
                            label="💾 Download Code",
                            data=st.session_state[f"user_code_{playground_key}"],
                            file_name="playground_code.py",
                            mime="text/x-python-script",
                            key=f"download_code_{playground_key}"
                        )
                        if st.session_state.get(playground_error_key):
                            st.error(f"Error running code: {st.session_state[playground_error_key]}")
                        elif st.session_state.get(playground_output_key) is not None:
                            st.success("Code executed successfully!")
                            st.markdown("**Playground Output:**")
                            result = st.session_state[playground_output_key]
                            if result:
                                for r in result:
                                    st.write(r)
                                    # --- Model Evaluation Visualizations ---
                                    if isinstance(r, pd.DataFrame):
                                        st.dataframe(r)
                                    elif isinstance(r, plt.Figure):
                                        st.pyplot(r)
                                    elif isinstance(r, str) and r.startswith("Confusion Matrix"):
                                        st.markdown(f"**{r}**")
                                    elif isinstance(r, str) and r.startswith("ROC Curve"):
                                        st.markdown(f"**{r}**")
                            else:
                                st.info("No output or error running the code.")
                        st.markdown("---")
        else:
            st.info("Please select features and target in the first tab.")


if __name__ == "__main__":
    main()