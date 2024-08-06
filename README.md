## install streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit 

## export package libraries
pip freeze > requirements.txt

python.exe -m pip install --upgrade pip

pip install -U ibm-watson-machine-learning
pip install langchain
pip install pypdf
pip install chromadb
pip install sentence_transformers
pip install langchain_community
pip install -U langchain-ibm
pip install torch==2.2.0

