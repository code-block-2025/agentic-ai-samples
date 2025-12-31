Pre-Requiste:

Install Extensions:
    Jupyter
    Python
    Python Debugger
    PowerShell

Install UV Python environment manager
    Windows CMD prompt : powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Clone the repository:  https://github.com/code-block-2025/agentic-ai-samples.git


Ensure you are on root path (Agentic-AI-Samples)

Install Python Virtual Environment. Run following commands in Terminal (VSCode/Cursor)
    uv python install 3.13.11
    uv venv --python 3.13.11 

Activate the Environment. Run following command in Terminal:
    .venv\Scripts\activate

Install required packages listed in pyproject.toml in virtual environment. Run following command(s) in Terminal
    uv pip install -r requirements.txt
    uv sycn

Open verify.ipynb notebook

Click Select Kernel button shown on right top of the notebook.

Select Python Environment 
Select venv 3.13.11

Hit run button on Note book cell. Verify Hello World is printed.

Simple LLM call is made provided your Open AI API Key is updated in .env file...

You can generate trial key at: 

https://platform.openai.com/


Documentation:

https://platform.openai.com/docs/overview






