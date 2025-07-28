import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
import numpy as np
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
                    <div style='color:#00c6ff;font-size:1.1em;font-family:sans-serif;font-weight:bold;text-shadow:0 1px 6px #ffd200;margin-top:8px;'>ML Model Builder</div>
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
            compare_models = st.multiselect("Models", ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "Linear Regression", "XGBoost", "LightGBM", "CatBoost"], default=[st.session_state.selected_model_type])
            st.info("Click 'Run Comparison' to train and compare selected models.")
            cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
            if st.button("Run Comparison"):
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, confusion_matrix, roc_curve, precision_recall_curve
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
                model_objs = {}
                import matplotlib.pyplot as plt
                import seaborn as sns
                for model_name in compare_models:
                    model = None
                    try:
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
                        elif model_name == "XGBoost":
                            try:
                                from xgboost import XGBClassifier, XGBRegressor
                                if y.nunique() > 10:
                                    model = XGBRegressor()
                                else:
                                    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                            except Exception as e:
                                st.warning(f"XGBoost not installed: {e}")
                                continue
                        elif model_name == "LightGBM":
                            try:
                                from lightgbm import LGBMClassifier, LGBMRegressor
                                if y.nunique() > 10:
                                    model = LGBMRegressor()
                                else:
                                    model = LGBMClassifier()
                            except Exception as e:
                                st.warning(f"LightGBM not installed: {e}")
                                continue
                        elif model_name == "CatBoost":
                            try:
                                from catboost import CatBoostClassifier, CatBoostRegressor
                                if y.nunique() > 10:
                                    model = CatBoostRegressor(verbose=0)
                                else:
                                    model = CatBoostClassifier(verbose=0)
                            except Exception as e:
                                st.warning(f"CatBoost not installed: {e}")
                                continue
                        else:
                            continue
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        metrics = {}
                        if y.nunique() > 10:
                            metrics['MSE'] = mean_squared_error(y_test, y_pred)
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
                            metrics['CV_MSE'] = -cv_scores.mean()
                        else:
                            metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                            metrics['F1'] = f1_score(y_test, y_pred, average='weighted')
                            try:
                                if hasattr(model, "predict_proba"):
                                    metrics['ROC AUC'] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo')
                                else:
                                    metrics['ROC AUC'] = 'N/A'
                            except Exception:
                                metrics['ROC AUC'] = 'N/A'
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                            metrics['CV_Accuracy'] = cv_scores.mean()
                        # Visualizations
                        st.markdown(f"### {model_name} Results")
                        if y.nunique() > 10:
                            st.metric("Test MSE", metrics['MSE'])
                            st.metric("CV MSE", metrics['CV_MSE'])
                        else:
                            st.metric("Test Accuracy", metrics['Accuracy'])
                            st.metric("CV Accuracy", metrics['CV_Accuracy'])
                            # Confusion Matrix
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm, ax_cm = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
                            ax_cm.set_title(f"Confusion Matrix: {model_name}")
                            st.pyplot(fig_cm)
                            # ROC Curve
                            if hasattr(model, "predict_proba"):
                                try:
                                    y_score = model.predict_proba(X_test)
                                    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1] if y_score.shape[1] > 1 else y_score)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr)
                                    ax_roc.set_title(f"ROC Curve: {model_name}")
                                    st.pyplot(fig_roc)
                                except Exception:
                                    pass
                            # PR Curve
                            try:
                                if hasattr(model, "predict_proba"):
                                    y_score = model.predict_proba(X_test)
                                    precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1] if y_score.shape[1] > 1 else y_score)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall, precision)
                                    ax_pr.set_title(f"PR Curve: {model_name}")
                                    st.pyplot(fig_pr)
                            except Exception:
                                pass
                        results.append({'Model': model_name, **metrics})
                        model_objs[model_name] = model
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
                leaderboard = pd.DataFrame(results)
                st.dataframe(leaderboard)
                # Highlight best model
                best_model_name = None
                if not leaderboard.empty:
                    if 'Accuracy' in leaderboard.columns:
                        best_idx = leaderboard['Accuracy'].idxmax()
                        best_model_name = leaderboard.loc[best_idx, 'Model']
                        st.success(f"Best Model: {best_model_name} (Accuracy: {leaderboard.loc[best_idx, 'Accuracy']:.3f})")
                    elif 'MSE' in leaderboard.columns:
                        best_idx = leaderboard['MSE'].idxmin()
                        best_model_name = leaderboard.loc[best_idx, 'Model']
                        st.success(f"Best Model: {best_model_name} (MSE: {leaderboard.loc[best_idx, 'MSE']:.3f})")
                st.download_button("Download Leaderboard", data=leaderboard.to_csv(index=False), file_name="leaderboard.csv")
                # --- Explain Model Choice (LLM) ---
                if best_model_name:
                    if st.button("Explain Model Choice"):
                        explain_prompt = f"Explain why {best_model_name} is the best model for this dataset and task. Compare with other models in the leaderboard."
                        with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                            _, explanation = chat_with_llm(code_interpreter, explain_prompt, f"./{st.session_state.uploaded_file_name}")
                            st.info(explanation)
        else:
            st.info("Upload a CSV and select features/target to compare models.")

    # --- Explainable AI (SHAP, LIME, Permutation Importance) Tab ---
    with tab6:
        st.header("🔍 Explainable AI (SHAP, LIME, Permutation Importance)")
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
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(plt.gcf())
            # --- LIME ---
            st.subheader("LIME Explanation")
            try:
                from lime.lime_tabular import LimeTabularExplainer
                lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=[str(c) for c in set(y_train)], discretize_continuous=True)
                sample_idx = st.number_input("Select row for LIME explanation", min_value=0, max_value=len(X_test)-1, value=0)
                exp = lime_explainer.explain_instance(X_test.values[sample_idx], model.predict_proba if hasattr(model, "predict_proba") else model.predict, num_features=5)
                st.write(exp.as_list())
            except Exception as e:
                st.warning(f"LIME not available: {e}")
            # --- Permutation Importance ---
            st.subheader("Permutation Importance")
            try:
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(model, X_test, y_test)
                perm_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm.importances_mean})
                st.dataframe(perm_df.sort_values('Importance', ascending=False))
            except Exception as e:
                st.warning(f"Permutation importance error: {e}")
            # --- Per-sample SHAP Explanation ---
            st.subheader("Per-sample SHAP Explanation")
            sample_idx = st.number_input("Select row for SHAP explanation", min_value=0, max_value=len(X_test)-1, value=0)
            shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx], matplotlib=True, show=False)
            st.pyplot(plt.gcf())
            # --- Downloadable Explanation Report ---
            st.subheader("Download Explanation Report")
            import io
            report = io.StringIO()
            report.write("SHAP Feature Importance\n")
            report.write(str(pd.DataFrame({'Feature': X_test.columns, 'Importance': np.abs(shap_values).mean(axis=0)})) + "\n\n")
            report.write("Permutation Importance\n")
            try:
                report.write(str(perm_df) + "\n\n")
            except:
                pass
            st.download_button("Download Explanation Report", data=report.getvalue(), file_name="explanation_report.txt")
            st.info("SHAP, LIME, and permutation importance show which features most influence model predictions. For more details, see documentation.")
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
            # --- Column Tooltips ---
            col_info = []
            for col in df.columns:
                missing_pct = 100 * df[col].isnull().sum() / len(df)
                unique_vals = df[col].nunique()
                col_info.append({"Column": col, "Type": df[col].dtype, "Missing %": f"{missing_pct:.1f}", "Unique": unique_vals})
            st.write("**Column Types & Info:**")
            st.dataframe(pd.DataFrame(col_info))
            # --- Data Type Conversion ---
            st.subheader("Convert Data Types")
            convert_col = st.selectbox("Select column to convert type", df.columns)
            convert_type = st.selectbox("Convert to type", ["numeric", "categorical", "string"])
            if st.button("Convert Type"):
                try:
                    if convert_type == "numeric":
                        df[convert_col] = pd.to_numeric(df[convert_col], errors="coerce")
                    elif convert_type == "categorical":
                        df[convert_col] = df[convert_col].astype("category")
                    elif convert_type == "string":
                        df[convert_col] = df[convert_col].astype(str)
                    st.success(f"Converted {convert_col} to {convert_type}.")
                except Exception as e:
                    st.error(f"Error converting: {e}")
            # --- Feature Selection Algorithms ---
            st.subheader("Feature Selection Algorithms")
            fs_method = st.selectbox("Select feature selection method", ["None", "SelectKBest", "Recursive Feature Elimination (RFE)"])
            fs_n_features = st.slider("Number of features to select", 1, max(1, len(df.columns)-1), 3)
            selected_features_algo = None
            if fs_method != "None" and st.session_state.selected_target:
                try:
                    from sklearn.feature_selection import SelectKBest, f_classif, RFE
                    from sklearn.linear_model import LogisticRegression
                    X_fs = df.drop(columns=[st.session_state.selected_target])
                    y_fs = df[st.session_state.selected_target]
                    # Encode categorical features
                    for col in X_fs.select_dtypes(include=['object']).columns:
                        X_fs[col] = X_fs[col].astype('category').cat.codes
                    if y_fs.dtype == 'object':
                        y_fs = y_fs.astype('category').cat.codes
                    if fs_method == "SelectKBest":
                        selector = SelectKBest(score_func=f_classif, k=fs_n_features)
                        selector.fit(X_fs, y_fs)
                        selected_features_algo = list(X_fs.columns[selector.get_support()])
                    elif fs_method == "Recursive Feature Elimination (RFE)":
                        model = LogisticRegression(max_iter=1000)
                        selector = RFE(model, n_features_to_select=fs_n_features)
                        selector.fit(X_fs, y_fs)
                        selected_features_algo = list(X_fs.columns[selector.get_support()])
                    st.info(f"Selected features: {selected_features_algo}")
                except Exception as e:
                    st.error(f"Feature selection error: {e}")
            # --- Feature and Target Selection ---
            st.subheader("Select Features and Target")
            columns = df.columns.tolist()
            target = st.selectbox("Target Column (what to predict)", columns, key="target_select")
            features_default = selected_features_algo if selected_features_algo else [col for col in columns if col != target][:min(3, len(columns)-1)]
            features = st.multiselect("Feature Columns (inputs)", [col for col in columns if col != target], default=features_default, key="features_select")
            st.session_state.selected_target = target
            st.session_state.selected_features = features
            # --- Save/Load Feature/Target Selection ---
            st.subheader("Save/Load Feature & Target Selection")
            if st.button("Save Selection"):
                try:
                    with open("feature_target_selection.json", "w") as f:
                        json.dump({"features": features, "target": target}, f)
                    st.success("Selection saved.")
                except Exception as e:
                    st.error(f"Save error: {e}")
            if st.button("Load Selection"):
                try:
                    with open("feature_target_selection.json", "r") as f:
                        sel = json.load(f)
                    st.session_state.selected_features = sel["features"]
                    st.session_state.selected_target = sel["target"]
                    st.success("Selection loaded.")
                except Exception as e:
                    st.error(f"Load error: {e}")
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
            # --- Interactive EDA Widgets ---
            st.subheader("Interactive EDA")
            eda_col = st.selectbox("Select column for EDA", df.columns)
            eda_type = st.selectbox("EDA Chart Type", ["Histogram", "Boxplot", "Violinplot", "Pairplot"])
            import seaborn as sns
            import matplotlib.pyplot as plt
            if eda_type == "Histogram":
                fig, ax = plt.subplots()
                sns.histplot(df[eda_col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
            elif eda_type == "Boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x=df[eda_col], ax=ax)
                st.pyplot(fig)
            elif eda_type == "Violinplot":
                fig, ax = plt.subplots()
                sns.violinplot(x=df[eda_col], ax=ax)
                st.pyplot(fig)
            elif eda_type == "Pairplot":
                fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
                st.pyplot(fig)
            # --- Smart Recommendations for Feature Engineering ---
            st.subheader("Smart Recommendations")
            recommendations = []
            if missing.sum() > 0:
                recommendations.append("Impute or drop columns with missing values.")
            corr = df.corr(numeric_only=True)
            high_corr = corr[(abs(corr) > 0.8) & (abs(corr) < 1.0)]
            if not high_corr.empty:
                recommendations.append("Remove or combine highly correlated features.")
            if any(df[col].nunique() < 5 for col in df.columns):
                recommendations.append("Consider one-hot encoding for categorical columns with few unique values.")
            if recommendations:
                st.info("\n".join(recommendations))
            else:
                st.success("No major feature engineering recommendations.")
            # --- Anomaly Detection ---
            st.subheader("Anomaly Detection")
            anomaly_method = st.selectbox("Choose anomaly detection method", ["None", "Isolation Forest", "DBSCAN"])
            if anomaly_method != "None":
                try:
                    from sklearn.ensemble import IsolationForest
                    from sklearn.cluster import DBSCAN
                    X_anom = df.select_dtypes(include=['float64', 'int64']).dropna()
                    if anomaly_method == "Isolation Forest":
                        clf = IsolationForest(contamination=0.05)
                        preds = clf.fit_predict(X_anom)
                        anomalies = X_anom[preds == -1]
                        st.write(f"Isolation Forest detected {anomalies.shape[0]} anomalies.")
                        st.dataframe(anomalies)
                    elif anomaly_method == "DBSCAN":
                        db = DBSCAN(eps=0.5, min_samples=5)
                        preds = db.fit_predict(X_anom)
                        anomalies = X_anom[preds == -1]
                        st.write(f"DBSCAN detected {anomalies.shape[0]} anomalies.")
                        st.dataframe(anomalies)
                except Exception as e:
                    st.error(f"Anomaly detection error: {e}")
            # --- Downloadable EDA Report ---
            st.subheader("Download EDA Report")
            import io
            eda_report = io.StringIO()
            eda_report.write("Missing Value Analysis\n")
            eda_report.write(str(missing) + "\n\n")
            eda_report.write("Correlation Matrix\n")
            eda_report.write(str(corr) + "\n\n")
            eda_report.write("Smart Recommendations\n")
            eda_report.write("\n".join(recommendations) + "\n\n")
            st.download_button("Download EDA Report", data=eda_report.getvalue(), file_name="eda_report.txt")
            # --- Outlier Detection (Z-score) ---
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
            # --- Feature Correlations ---
            st.subheader("Feature Correlations")
            st.dataframe(corr)
            st.markdown("**Highly correlated features (>|0.8|):**")
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
            chart_types = ["Bar", "Scatter", "Box", "Heatmap", "Pie", "Line", "Violin", "Pairplot"]
            chart_type = st.selectbox("Chart Type", chart_types)
            x_col = st.selectbox("X Axis", df.columns)
            y_col = st.selectbox("Y Axis", [col for col in df.columns if col != x_col])
            # --- Chart Customization ---
            color_col = st.selectbox("Color (optional)", [None] + list(df.columns))
            agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "none"])
            chart_size = st.slider("Chart Size", 4, 12, 6)
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig = None
            if chart_type == "Bar":
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    if agg_func != "none":
                        bar_df = df[[x_col, y_col]].groupby(x_col)[y_col].agg(agg_func).reset_index()
                    else:
                        bar_df = df[[x_col, y_col]].copy()
                    fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                    sns.barplot(x=x_col, y=y_col, data=bar_df, ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning(f"Column '{y_col}' is not numeric. Please select a numeric column for Y axis.")
            elif chart_type == "Scatter":
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                sns.scatterplot(x=df[x_col], y=df[y_col], hue=df[color_col] if color_col else None, ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
            elif chart_type == "Box":
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                sns.boxplot(x=df[x_col], y=df[y_col], hue=df[color_col] if color_col else None, ax=ax)
                st.pyplot(fig)
            elif chart_type == "Heatmap":
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            elif chart_type == "Pie":
                pie_data = df[x_col].value_counts()
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                st.pyplot(fig)
            elif chart_type == "Line":
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                sns.lineplot(x=df[x_col], y=df[y_col], ax=ax)
                st.pyplot(fig)
            elif chart_type == "Violin":
                fig, ax = plt.subplots(figsize=(chart_size, chart_size))
                sns.violinplot(x=df[x_col], y=df[y_col], ax=ax)
                st.pyplot(fig)
            elif chart_type == "Pairplot":
                fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
                st.pyplot(fig)
            # --- Interactive Filtering ---
            st.subheader("Filter Data")
            filter_col = st.selectbox("Filter Column", df.columns)
            filter_type = st.selectbox("Filter Type", ["Exact", "Range", "Multi-select"])
            filtered_df = df.copy()
            if filter_type == "Exact":
                filter_val = st.text_input("Filter Value (exact match)")
                if filter_val:
                    filtered_df = filtered_df[filtered_df[filter_col].astype(str) == filter_val]
            elif filter_type == "Range":
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    min_val, max_val = st.slider("Select range", float(df[filter_col].min()), float(df[filter_col].max()), (float(df[filter_col].min()), float(df[filter_col].max())))
                    filtered_df = filtered_df[(filtered_df[filter_col] >= min_val) & (filtered_df[filter_col] <= max_val)]
            elif filter_type == "Multi-select":
                options = st.multiselect("Select values", df[filter_col].unique())
                if options:
                    filtered_df = filtered_df[filtered_df[filter_col].isin(options)]
            st.dataframe(filtered_df)
            # --- Export to PNG/JPG ---
            st.subheader("Export Chart")
            if fig is not None:
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button("Download Chart as PNG", data=buf.getvalue(), file_name="chart.png")
            st.download_button("Download Chart Data", data=filtered_df.to_csv(index=False), file_name="chart_data.csv")
        else:
            st.info("Upload a CSV file to visualize data.")

    with tab2:
        # ...existing code...
        if df is not None and st.session_state.selected_features and st.session_state.selected_target:
            st.subheader("AI Model Builder & Code Playground")
            # --- Example Prompts/Templates ---
            st.markdown("#### Example ML Prompts")
            example_prompts = {
                "Classification": "Train a Random Forest classifier to predict the target using selected features.",
                "Regression": "Train a Linear Regression model to predict the target using selected features.",
                "Feature Importance": "Show feature importances for the selected model.",
                "Confusion Matrix": "Plot the confusion matrix for the trained model.",
            }
            selected_example = st.selectbox("Choose example prompt", list(example_prompts.keys()))
            if st.button("Insert Example Prompt"):
                st.session_state.example_prompt = example_prompts[selected_example]
            ml_prompt = st.session_state.get("example_prompt", "")
            # --- Build ML Prompt with Feature Engineering and Hyperparameters ---
            ml_prompt += f"\nYou are a Python machine learning expert. Given a dataset at path './{st.session_state.uploaded_file_name}', write code to:\n- Load the CSV\n- Apply feature scaling: {st.session_state.selected_scaler}\n- Apply categorical encoding: {st.session_state.selected_encoder}\n- Split into train/test\n- Train a {st.session_state.selected_model_type} model to predict '{st.session_state.selected_target}' using features {st.session_state.selected_features}\n- Use these hyperparameters: "
            if st.session_state.selected_model_type == "Random Forest":
                ml_prompt += f"n_estimators={st.session_state.n_estimators}, max_depth={st.session_state.max_depth}\n"
            elif st.session_state.selected_model_type == "Decision Tree":
                ml_prompt += f"max_depth={st.session_state.max_depth}\n"
            elif st.session_state.selected_model_type == "Support Vector Machine":
                ml_prompt += f"C={st.session_state.c_value}\n"
            ml_prompt += f"- Print train/test accuracy or relevant metrics\n- Show feature importances if available\n- Plot confusion matrix, ROC curve, and other relevant evaluation visualizations\nIMPORTANT: Always use the dataset path variable './{st.session_state.uploaded_file_name}' in your code.\n"
            st.write("#### AI will generate code for your selected ML task:")
            st.code(ml_prompt)
            # --- Explain Code Button ---
            if st.button("Explain Code"):
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    explain_prompt = f"Explain the following Python code step by step:\n{ml_prompt}"
                    _, explanation = chat_with_llm(code_interpreter, explain_prompt, f"./{st.session_state.uploaded_file_name}")
                    st.info(explanation)
            # --- Generate & Run ML Code ---
            if st.button("Generate & Run ML Code"):
                # --- Code Validation ---
                import ast
                try:
                    ast.parse(ml_prompt)
                except Exception as e:
                    st.error(f"Syntax error in generated code: {e}")
                    return
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
                                    # --- Code Validation Before Running ---
                                    try:
                                        ast.parse(st.session_state[f"user_code_{playground_key}"])
                                    except Exception as e:
                                        st.session_state[playground_error_key] = f"Syntax error: {e}"
                                    else:
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
                        # --- Output Visualization ---
                        if st.session_state.get(playground_error_key):
                            st.error(f"Error running code: {st.session_state[playground_error_key]}")
                        elif st.session_state.get(playground_output_key) is not None:
                            st.success("Code executed successfully!")
                            st.markdown("**Playground Output:**")
                            result = st.session_state[playground_output_key]
                            if result:
                                for r in result:
                                    if isinstance(r, pd.DataFrame):
                                        st.dataframe(r)
                                    elif isinstance(r, plt.Figure):
                                        st.pyplot(r)
                                    elif isinstance(r, str) and r.startswith("Confusion Matrix"):
                                        st.markdown(f"**{r}**")
                                    elif isinstance(r, str) and r.startswith("ROC Curve"):
                                        st.markdown(f"**{r}**")
                                    elif isinstance(r, (int, float)):
                                        st.metric("Metric Value", r)
                                    else:
                                        st.write(r)
                            else:
                                st.info("No output or error running the code.")
                        st.markdown("---")
        else:
            st.info("Please select features and target in the first tab.")


if __name__ == "__main__":
    main()