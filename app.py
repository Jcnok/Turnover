from os import sep
from git import Git
import pandas as pd
import numpy as np 
import streamlit as st
from pycaret.classification import load_model, predict_model
from pycaret import *
import PIL
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pycaret.utils import check_metric
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
ico = Image.open('ico3.png')
#ico = Image.open('2103589.ico')
st.set_page_config(
    page_title="Nihontech",
    page_icon= ico,    
    layout="wide", #centered",
    initial_sidebar_state='auto',
    menu_items=None)
paginas = ['Home','Case1','Análise de Turnover', 'Case2', "Demonstação", "Predição de turnover", "Dashbord comparativo"]
site = "https://app.powerbi.com/view?r=eyJrIjoiZmM5MDAwMDAtYmM3Mi00MjE5LTllZTUtNDcxYjY3NTVjMDAxIiwidCI6Ijk4ZjkwMzVmLTZkOWMtNDBmMy1hNDI0LWI0NDY0M2NjMmYyZiJ9&embedImagePlaceholder=true"
site_pred = "https://app.powerbi.com/view?r=eyJrIjoiZTFhYjYyNWYtMTk4ZS00NGViLWFkM2YtNzVlMDM4NGM4NTc4IiwidCI6Ijk4ZjkwMzVmLTZkOWMtNDBmMy1hNDI0LWI0NDY0M2NjMmYyZiJ9"
###### SIDE BAR ######
col1, col2, col3 = st.sidebar.columns([0.5, 1, 1])
with col2:
    image1 = Image.open('logo_size.jpg')
    st.image(image1, width=120)
    pagina = st.sidebar.selectbox("Navegação", paginas)
###### PAGINA INICIAL ######
if pagina == 'Home':
    lottie_1 = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_3FIGvm.json')
    st_lottie(lottie_1, speed=0.3, height=150, key="initial")
    st.subheader("Análise de turnover")    
    st.write("""
    
    Aproveitando os estudos e o aprendizado adquirido durante o campeonato de Machine Learning promovido no curso da Flai,
    resolvi aproveitar a base de dados, para criar esse projeto visando como seria em um cenário próximo do real, trabalhando como consultor.
    
    O projeto envolve a criação de um Dashboard desenvolvido no Power bi e o desenvolvimento de um modelo de classificação para predição de Turnover.
    
    O objetivo aqui é demonstrar os benefícios que essas tecnologias podem oferecer dentro da visão do negócio, 
    procurei não utilizar termos técnicos e nem a parte de tratamento e técnicas utilizadas para criação do projeto, caso tenham interesse, 
    todos os arquivos podem ser acessados no meu [GitHub](https://github.com/Jcnok/turnover).
        
    """)
###### PAGINA CASE1 ######
if pagina == 'Case1':
    lottie_2 = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_m9zragkd.json')
    st_lottie(lottie_2, speed=0.5, height=150, key="initial")
    st.subheader("Problema do Negócio")
    
    st.write("""

    Case: O Empresário Ricardo, dono de várias empresas entrou em contato, pois está com sérios problemas com o quadro de funcionários.

    Muitos funcionários estão deixando suas empresas já no primeiro ano e isso está causando sérios problemas financeiros por falta de Mão de obra qualificada.
    
    Durante nossa primeira reunião, o empresário informa que deseja ter uma visão mais ampla com algumas métricas importantes em uma única tela,
        para que possa embasar suas decisões e acompanhar o desempenho de suas empresas de forma rápida e eficaz.
""")


    st.write("""
    Diante desse cenário, propomos a criação de um dashboard no Power bi, que por sinal é parte do curso da Flai e temos todos os quesitos para realizá-lo.

    Foram levantadas todas as perguntas de negócio a serem respondidas e todas as métricas necessárias para o projeto.

    Já de acordo com a LGPD o cliente irá fornecer metade da base de dados dos funcionários de forma anônima e aleatória. 
    
    O resultado do projeto pode ser conferido no menu lateral em Análise de Turnover.        
""")
    


###### BI ######
if pagina == 'Análise de Turnover':
    st.subheader("Análise de Turnover")    
    col1,col2,col3 = st.columns([1,2,3])
    st.components.v1.iframe(site, width=1400, height=800, scrolling=True)

    st.sidebar.write("""O Dashbord é interativo basta posicionar o mouse sobre os gráficos para obter a quantidade de funcionários,
    a média salarial e o turnover por segmento.   
    
    """)
###### PAGINA CASE2 ######
if pagina == 'Case2':
    lottie_3 = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_q5qeoo3q.json')
    st_lottie(lottie_3, speed=0.5, height=150, key="initial")
    st.subheader("Proposta de solução usando Machine Learning")
    st.write("""
    
        O cliente ficou muito satisfeito com o dashboard, agora ele pode rapidamente obter as informações desejadas para tomada de decisões.
        
        Podemos perceber por exemplo, que funcionários com ensino superior tende a dar um maior turnover, possivelmente deva receber várias propostas por sua qualificação,
        o sexo feminino apesar de representar um quadro menor de funcionários tende a dar um turnover maior, as empresas com maior quadro de funcionários também estão
        propensas à uma maior taxa, são alguns insights que podemos ter apenas ao visualizar os gráficos, porém cabe ao cliente tirar as suas devidas
        conclusões para tomar as decisões de melhorias para redução de turnover.

        Durante uma nova reunião o cliente informou que irá aportar recursos financeiros para retenção e contratação de novos funcionários visando reduzir a alta taxa de turnover.
        Como um bom consultor fiz uma pergunta ao cliente. E se fosse possível lhe dar uma lista dos funcionários com maior propensão ao turnover para que possa trabalhar de forma bem objetiva e reduzir assim gastos desnecessários, não seria interessante? Poderia por exemplo passar essa lista por região para que os gerentes possam criar suas devidas estratégias e após um teste AB por exemplo, aplicar a melhor medida. Durante as contratações poderíamos criar um perfil de funcionário mais adequado para cada setor que seja menos propenso ao turnover, as possibilidades são inúmeras.
    """)

    st.write("""
	    Parabéns, se chegou até aqui, bem-vindos ao Machine Learning!

    Case 02: Nosso cliente ficou muito interessado, mas com uma certa desconfiança ele nos propôs o seguinte:  
    Como fiquei com a metade do banco de dados dos funcionários, na próxima reunião,
    quero uma demonstração do quanto essa tecnologia será capaz de prever os funcionários que deixaram a empresa,
    baseado no conjunto de dados que vocês ainda não sabem, e diante do resultado disso, podemos fechar ou não esse novo contrato de consultoria.


    Aceitamos o desafio, o projeto pode ser conferido no menu Lateral em “Demonstração”.    
    """)    

###### Demonstação do modelo de machine learning ######
if pagina == 'Demonstação':
    st.sidebar.write("""Nesse exemplo: o cliente irá carregar a outra parte da base de dados que não possuímos, essa base deve estar no mesmo formato da primeira base.
    obs.: caso desejado poderá utilizar o arquivo 'teste_base.csv' basta copiar esse link: [teste_base.csv](https://raw.githubusercontent.com/Jcnok/turnover/main/teste_base.csv) ao clicar em Browse files cole o caminho e clique em abrir.
     
    """)

    st.markdown("### Carregue a base de dados no formato .csv contendo o restante da base de dados dos funcionários")
    st.markdown("---")
    uploaded_file = st.file_uploader("escolha o arquivo *.csv")
    if uploaded_file is not None:
        dados = pd.read_csv(uploaded_file, sep=';', decimal=',')
        #dados = pd.read_csv(uploaded_file)
        st.write(dados.head()) # checar a saída no terminal
    
    if st.button('CLIQUE AQUI PARA EXECUTAR O MODELO'):
        modelo = load_model('./comb_soft_model') 
        pred = predict_model(modelo, data = dados)        
        classe_sim = pred.query('turnover_apos_1_ano == "SIM"')
        classe_label_sim = pred.query('Label == "SIM"')['Label'].count()
        count_pred = (pred['Label'] == 'SIM').sum()
        count_pred_Label = (classe_sim['Label']=="SIM").sum()
        count_total = (classe_sim['turnover_apos_1_ano']).count()
        recal = check_metric(pred['turnover_apos_1_ano'], pred['Label'], metric='Recall')
        result = f'''
        Em um total de {classe_label_sim} chutes, o modelo foi capaz de identificar {count_pred_Label} funcionários, dos {count_total} que realmente saíram por algum motivo. Dentro da lista, o modelo encontrou {round((recal * 100),2)}% de todos os funcionários que deram turnover. Lembrando que apesar de um bom resultado, essa é apenas uma demonstração. Podemos facilmente melhorar essa precisão.'''
        st.subheader(result)
        st.markdown("---")
        st.markdown('### Caso desejado, você pode realizar o download do resultado no formato .csv clicando logo abaixo!')
        #st.write(pred.query('Label == "SIM"')[['func_sexo','Score']].sort_values('Score', ascending=False))
        #pred.to_csv('dados_preditos.csv',sep=';', decimal=',') # Salvando o modelo para incluí-lo no powerbi.
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun        
            return df.to_csv(sep=';',decimal=',').encode('utf-8')               
        csv = convert_df(pred)    
        st.download_button(
        label="Download do aquivo .CSV",
        data=csv,
        file_name='predict.csv',
        mime='text/csv',
        )
      

###### Modelo de predição ######
if pagina == 'Predição de turnover':
    st.markdown('### Selecione as opções de acordo com os dados do funcionário e execute o modelo!')
    st.sidebar.write("""Aqui o cliente consegue selecionar os dados do perfil do funcinário de forma individual.
     O modelo irá informar se esse perfil tem ou não uma tendência maior ao turnover em menos de 1 ano.
    """)

    st.markdown('---')    
    sexo = st.radio('Selecione o Sexo',['MASCULINO', 'FEMININO'])
    idade = np.int64(st.slider('Entre com a idade:', 16, 85, 20))	
    raca = st.selectbox('Entre com a raça:',['AMARELA-BRANCA', 'SEM INFO', 'PRETA-PARDA', 'INDIGENA'])
    escolaridade = st.selectbox('Entre com a escolaridade:',['MEDIO COMPLETO', 'ANALFABETO-FUND_INCOMPLETO','SUPERIOR_COMPLETO', 'FUND_COMPLETO-MEDIO_INCOMPLETO','SUPERIOR_INCOMPLETO'])
    uf = st.selectbox('Entre com a uf:', ['RN', 'BA', 'SE', 'PE', 'CE', 'MA', 'PI', 'PB', 'AL'])
    deficiente = st.radio('Deficiente:', ['SIM','NAO'])
    empresa_porte = st.selectbox('Seleciona o porte da empresa:', ['DE 10 A 19', 'ATE 4', 'DE 20 A 49', 'DE 5 A 9', '1000 OU MAIS','DE 500 A 999', 'DE 50 A 99', 'DE 100 A 249', 'DE 250 A 499'])
    setor = st.selectbox('Selecione o setor:', ['Serviços', 'Comércio', 'Construçao civil', 'Agricultura','Indústria', 'Administraçao pública'])
    tb_horas = np.int64(st.slider('Selecione tipo de contrato em horas por semana:', 0,44,40)) 
    salario = np.float64(st.slider('Selecione o Salário:',200,120000, 1950))

    st.markdown('---')
        
    dic = {'func_sexo': [sexo], 'func_idade': [idade], 'func_racacor': [raca],'func_escolaridade': [escolaridade],
            'func_uf': [uf], 'func_deficiencia': [deficiente],'empresa_porte':[empresa_porte],'empresa_setor':[setor],
            'contrato_horastrabalho': [tb_horas], 'contrato_salario':[salario]}  
    teste = pd.DataFrame(dic)
        


    if st.button('CLIQUE AQUI PARA EXECUTAR O MODELO'):
        modelo = load_model('./comb_soft_model') 
        pred_test = predict_model(modelo, data = teste)
        prob = list(pred_test.Score.round(2)*100)
        if pred_test.Label.values == "SIM":            
            result =f'''
                    Com uma probabilidade de {prob}%, o modelo identificou que esse funcionário tem uma maior tendência a deixar a firma em menos de 1 ano.
                    '''
        else:
           result =f'''
                    Com uma probabilidede de {prob}%, o modelo identificou que esse funcionário tem uma maior tendência em permanecer na empresa por mais de 1 ano                    ''' 
      
        st.subheader(result)
###### Dashboard Compartivo ######
if pagina == 'Dashbord comparativo':    
    st.subheader("Dashboard compartivo entre o resultado real Vs resultado do modelo")    
    col1,col2,col3 = st.columns([1,2,3])
    st.components.v1.iframe(site_pred, width=1400, height=800, scrolling=True)

    st.sidebar.write("""O Dashbord é interativo, posicione o mouse sobre os gráficos para obter o comparativo de acertos, entre turnover e a quantidade de demissões,
    por segmento.    
    """)
        
        

        