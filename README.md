# Dropout Insight: Educational Risk Dashboard with Counterfactual Explanations

Interactive tool to upload datasets, automatically train a machine learning model or reuse one that is already trained, and visualize and interpret the risk of dropout in education.

The application also integrates Retrieval-Augmented Generation and Large Language Models to transform individual and group counterfactual explanations into prescriptive educational recommendations supported by academic literature.

## 🚀 Principal characteristics

* Upload files in `.csv` format
* Automatic training of classification models
* Reuse of previously generated models
* Intuitive visualization of predictions and risk factors
* Model explanations and counterfactuals generation
* Individual counterfactual analysis
* Group counterfactual analysis
* Generation of prescriptive educational recommendations
* Retrieval of relevant educational literature using RAG
* Integration with different Large Language Models
* Selection between different recommendation generation modes
* Interactive dashboard developed in Plotly Dash

## 🤖 Generative AI functionality

The application includes a generative module that creates prescriptive recommendations from the results obtained by the predictive and counterfactual models.

The currently supported Large Language Models are:

* Google Gemini 2.5 Flash
* Llama 3.3 70B Versatile through Groq

The user can select between three generation modes:

* **Literature and counterfactuals:** uses the generated counterfactual explanations together with academic literature retrieved by the RAG system.
* **Counterfactuals only:** generates recommendations using only the changes identified in the counterfactual explanations.
* **Literature only:** generates recommendations using only the educational literature retrieved by the system.

The recommendations can be generated for both individual students and groups of students.

## 🛠️ Recommended virtual environment

Using a virtual environment is recommended to install dependencies in isolation.

Python 3 systems:

```bash
# Create virtual environment
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
```

```bash
# Linux or macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 API configuration

To use the generative recommendation module, API keys for the available LLM providers must be configured.

Create a `.env` file in the root directory of the project:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

The `.env` file must not be uploaded to GitHub.

Make sure it is included in `.gitignore`:

```gitignore
.env
```

## 📚 Build the RAG index

The RAG module uses academic publications related to educational dropout, student retention and educational interventions.

Before generating recommendations using literature, build the vector database by running:

```bash
python -m src.rag.build_rag_index
```

Alternatively, from inside the `src` directory:

```bash
cd src
python -m rag.build_rag_index
```

The index only needs to be rebuilt when new academic documents are added or the existing documents are modified.

## ▶️ Initialize the app

Run the application with:

```bash
python src/index.py
```

Access the dashboard at:

```text
http://localhost:8050
```

## 🖥️ Use

1. Select a dataset in `.csv` format.

2. The system checks if a saved model exists:

   * If it does not exist, the model is trained automatically with the uploaded data. A folder is created if needed at:

```text
src/saved_models/file_name
```

* If it exists, the saved model is loaded and used to generate predictions.

3. Navigate through the tabs available in the dashboard:

   * AutoML Report
   * Classification statistics
   * Features importance
   * What If Analysis
   * Counterfactuals
   * Group Counterfactuals

4. In the individual or group counterfactual sections:

   * Generate the counterfactual explanations.
   * Select the Large Language Model.
   * Select the recommendation generation mode.
   * Generate the prescriptive educational recommendation.

The generated response presents an interpretation of the identified risk factors and proposes possible educational actions.

When the literature mode is enabled, the recommendations are also supported by fragments retrieved from the academic document collection.

## 📦 Running the project with Docker

This project includes a Dockerfile that allows running the app without installing Python or dependencies manually.

### 1. Build the Docker image

The Dockerfile is located inside the `src/` folder.

From the root of the repository run:

```bash
docker build -t dropout-insight -f src/Dockerfile .
```

### 2. Run the container

Start the app:

```bash
docker run -p 8050:8050 dropout-insight
```

If API keys are required, they can be provided using an environment file:

```bash
docker run --env-file .env -p 8050:8050 dropout-insight
```

After running, open in your browser:

```text
http://localhost:8050
```

## ⭐ Fast option to automatically open the browser

This repository includes optional helper scripts that start the container and automatically open the browser.

### Windows: `run.bat`

```bat
@echo off
start "" http://localhost:8050
docker run --env-file .env -p 8050:8050 dropout-insight
```

Usage: double click `run.bat`.

### Linux or macOS: `run.sh`

```bash
#!/bin/bash
docker run --env-file .env -p 8050:8050 dropout-insight &
sleep 3
xdg-open "http://localhost:8050" 2>/dev/null || open "http://localhost:8050"
```

Usage:

```bash
chmod +x run.sh
./run.sh
```

### 3. Using local datasets inside Docker

Linux or macOS:

```bash
docker run --env-file .env -p 8050:8050 -v "$(pwd)/data:/app/data" dropout-insight
```

Windows PowerShell:

```powershell
docker run --env-file .env -p 8050:8050 -v ${PWD}\data:/app/data dropout-insight
```

### 4. Useful Docker commands

```bash
docker ps
docker stop <id>
docker rm <id>
docker rmi dropout-insight
```

## 📂 Repository structure

```text
├── src/
│   ├── rag/                    # RAG and LLM recommendation module
│   ├── tabs/                   # Dashboard tabs
│   ├── assets/                 # CSS styles
│   ├── saved_models/           # Trained models
│   ├── index.py                # Main application
│   └── Dockerfile
├── data/                       # Used datasets
├── doc/                        # User manual and documentation
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md                    # Project summary
```

## 📖 Additional documentation

The complete user manual is located in:

```text
doc/UserManual.pdf
```

## ⚠️ Disclaimer

The generated predictions and recommendations are intended to support educational decision-making.

They should not replace the assessment of teachers, academic advisers or other qualified professionals.

The application has been developed as an academic prototype and should not be used to make automatic decisions about students without human supervision.
