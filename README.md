# AI Receipt Parser

An intelligent, end-to-end system for extracting structured information from scanned receipt images using OCR `Tesseract`, transformer-based model `LayoutLMv3`, and LLM-powered reasoning `Groq/openai-oss-120b`.

## Features

- **OCR Extraction:** Uses Tesseract to extract text and bounding boxes from receipt images.
- **Vision Model Inference:** Fine-tuned LayoutLMv3 transformer (pushed to [Hugging Face](https://huggingface.co/Sameed1/smdk-layoutlmv3-receipts)) for key field extraction (company, address, date, total).
- **LLM Reasoning:** Integrates openai-oss-120b via Groq for advanced validation, correction, and reasoning over extracted fields.
- **Modular API:** Asynchronous FastAPI backend with clear separation of concerns (OCR, vision, LLM, config) and app lifespan configuration.
- **Prompt Engineering:** Agent's system prompt consisting of few shot prompting technique to yield correct results.

## Tech Stack

- **Python**
- **FastAPI** (API backend)
- **Uvicorn** (ASGI server)
- **Tesseract OCR** (text extraction)
- **Hugging Face Transformers** (LayoutLMv3)
- **Groq LLM API** (reasoning and validation)
- **Pydantic** (data validation)

## Project Structure

```
reciept parser/
├── server.py
├── inference/
│   ├── agent.py
│   ├── models.py
│   ├── routes.py
├── configs/
│   ├── groq_config.py
│   ├── huggingface_config.py
│   ├── tesseract_config.py
├── prompts/
│   ├── human_prompt.txt
│   ├── system_prompt.txt
```

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/Sameed-Khatri/AI-Receipt-Parser.git
cd AI-Receipt-Parser
```

### 2. Install Dependencies

Create a virtual environment and install required packages:

```sh
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 3. Configure Environment

- Set up your `.env` file with API keys and tesseract path.
```sh
GROQ_API_KEY = <API-KEY>
TESSERACT_PATH = <ABSOLUTE-PATH\tesseract.exe>
HF_API_TOKEN = <API-KEY>
```
- Ensure Tesseract is installed and available in your system.

### 4. Run the Server

```sh
python server.py
```

The API will be available at `http://localhost:8089`.

### 5. Run the Streamlit app

```sh
streamlit run streamlit.py
```