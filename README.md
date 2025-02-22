##Setup Environment - Shell/Terminal##
mkdir submissons_analisis
cd submissons_analisis
pipenv install
pipenv shell
pip install -r requirements.txt

##Run steamlit app##
streamlit run app.py