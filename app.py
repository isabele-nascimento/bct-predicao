"""Research demonstration; inputs live only in the current Streamlit session."""
from pathlib import Path
import streamlit as st
from inference import Predictor

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='BCT / UFRN | Desfechos academicos', layout='wide', initial_sidebar_state='collapsed')
st.html('''<style>
.stApp {background:#fafbfb;color:#202629;}
.block-container {max-width:1120px;padding-top:3.8rem;padding-bottom:3rem;}
div[data-testid="stCaptionContainer"],div[data-testid="stCaptionContainer"] * {color:#545e62!important;}
h1 {font-family:Georgia,serif!important;font-size:2rem!important;font-weight:400!important;letter-spacing:0!important;}
h2,h3 {letter-spacing:0!important;font-size:1.2rem!important;}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-testid="stNumberInputContainer"],
button[kind="primary"],button[kind="secondary"],div[data-testid="stAlert"] {border-radius:2px!important;}
div[data-testid="stAlert"] > div {border-radius:2px!important;}
div[data-testid="stMetricValue"] {font-family:Georgia,serif;font-size:2.5rem;}
div[data-testid="stTabs"] button {border-radius:0!important;}
@media(max-width:640px) {.block-container {padding-top:3.8rem;padding-left:1rem;padding-right:1rem;} h1 {font-size:1.7rem!important;}}
</style>''')


@st.cache_resource
def load_predictor():
    return Predictor()


try:
    predictor = load_predictor()
except (ValueError, OSError) as error:
    st.error('Modelo indisponível. Os arquivos de inferência precisam ser conferidos.')
    st.stop()

LABELS = {
    'F':'Feminino', 'M':'Masculino', 'SIM':'Sim', 'NAO':'Não', 'NAO_INFORMADO':'Não informado',
    'AMARELO (DE ORIGEM ORIENTAL)':'Amarelo (categoria original)', 'BRANCO':'Branco',
    'INDIGENA':'Indígena', 'NEGRO':'Negro (categoria original)', 'PARDO':'Pardo',
    'REMANESCENTE DE QUILOMBO':'Remanescente de quilombo',
    'ACORDO TIMOR-LESTE':'Acordo Timor-Leste', 'CONVENIO PEC-G':'Convênio PEC-G',
    'JUDICIAL':'Judicial', 'REOCUPACAO DE VAGAS RESIDUAIS':'Reocupação de vagas residuais',
    'SISU':'SiSU', 'TRANSF. COMPULSORIA':'Transferência compulsória',
    'EM ESCOLA PARTICULAR, COM BOLSA':'Particular, com bolsa', 'OUTRO TIPO DE ESCOLA':'Outro tipo de escola',
    'PARTE EM ESCOLA PUBLICA PARTE EM ESCOLA PARTICULAR':'Parte pública, parte particular',
    'TODO EM ESCOLA PARTICULAR':'Integralmente particular', 'TODO EM ESCOLA PUBLICA':'Integralmente pública',
}
FEATURE_LABELS = dict(sexo='sexo',raca='raça/cor',cotista='cota',forma_ingresso='forma de ingresso',
    escola='escola de origem',idade_ingresso='idade',log_renda='renda',media_notas='média das notas',
    media_faltas='média das faltas',disciplinas_avaliadas='disciplinas avaliadas',numero_reprovacoes='reprovações',
    proporcao_reprovacoes='proporção de reprovações',proporcao_reprovacoes_faltas='proporção de reprovações por faltas')


def category(label, name, help=None):
    index = predictor.schema['categorical_columns'].index(name)
    values = [v for v in predictor.schema['categories'][index] if v != 'NAO_INFORMADO']
    return st.selectbox(label, [None]+values, key=name,
        format_func=lambda v: 'Não informado' if v is None else LABELS.get(v,v), help=help)


def clear():
    for key in list(st.session_state):
        if key != 'clear':
            del st.session_state[key]


st.caption('PESQUISA EM TRAJETÓRIAS ACADÊMICAS')
st.title('BCT / UFRN')
st.write('Perfil, primeiro semestre e desfechos acadêmicos')
st.caption('Demonstração retrospectiva de pesquisa. Cancelamento versus conclusão observados; não é um risco futuro validado.')
simulation, methods = st.tabs(['Simulação', 'Modelo e avaliação'])

with simulation:
    left, right = st.columns([1.8, 1], gap='large')
    with left:
        with st.form('profile_form', border=False):
            st.subheader('Perfil de ingresso')
            c1,c2 = st.columns(2)
            with c1:
                sex=category('Sexo registrado','sexo')
                race=category('Raça/cor registrada','raca',help='Categorias originais da UFRN. Não foram convertidas em outras classificações.')
                school=category('Escola no ensino médio','escola')
                age=st.number_input('Idade no ano de ingresso',min_value=14,max_value=90,value=None,step=1,key='idade_ingresso',
                    placeholder='Não informada',help='Aproximação usada no estudo: ano de ingresso menos ano de nascimento.')
            with c2:
                quota=category('Ingresso por cota','cotista')
                entry=category('Forma de ingresso','forma_ingresso')
                income=st.number_input('Renda do discente: valor registrado',min_value=0.,max_value=1e9,value=None,step=100.,key='renda',
                    placeholder='Não informada',help='Campo renda da base original. O dicionário não esclarece a periodicidade; não foi equiparado a renda familiar ou salários mínimos. Não há atualização monetária.')
            st.divider()
            st.subheader('Semestre de ingresso')
            st.caption('Somente componentes com aprovação ou reprovação registrada. Trancamentos e cancelamentos de disciplinas não entram nas contagens.')
            a1,a2=st.columns(2)
            with a1:
                subjects=st.number_input('Disciplinas avaliadas',min_value=1,max_value=100,value=None,step=1,key='disciplinas_avaliadas',placeholder='Obrigatório')
                failed=st.number_input('Reprovações totais',min_value=0,max_value=100,value=None,step=1,key='numero_reprovacoes',placeholder='Obrigatório')
                failed_absence=st.number_input('Reprovações por faltas',min_value=0,max_value=100,value=None,step=1,key='reprovacoes_faltas',placeholder='Obrigatório',
                    help='Parte das reprovações totais, não uma contagem adicional.')
            with a2:
                grades=st.number_input('Média das notas finais',min_value=0.,max_value=10.,value=None,step=.1,key='media_notas',placeholder='Não informada',
                    help='Média aritmética das notas disponíveis nos componentes avaliados. Não é o CR.')
                absences=st.number_input('Média de faltas por disciplina',min_value=0.,max_value=1e5,value=None,step=.5,key='media_faltas',placeholder='Não informada',
                    help='Média do número total de faltas registrado nos componentes avaliados, não percentual.')
            submitted=st.form_submit_button('Calcular estimativa',type='primary',icon=':material/calculate:')
        st.button('Limpar',icon=':material/restart_alt:',on_click=clear,key='clear')
        st.caption('Sem nome, matrícula ou CPF. O aplicativo não grava os perfis preenchidos; os valores são processados no servidor durante a sessão.')
        if submitted:
            profile=dict(sexo=sex,raca=race,escola=school,idade_ingresso=age,cotista=quota,forma_ingresso=entry,renda=income,
                disciplinas_avaliadas=subjects,numero_reprovacoes=failed,reprovacoes_faltas=failed_absence,media_notas=grades,media_faltas=absences)
            try:
                st.session_state['result']=predictor.predict(profile)
            except ValueError as error:
                st.session_state.pop('result',None)
                st.error(str(error))
    with right:
        st.subheader('Estimativa do modelo')
        result=st.session_state.get('result')
        if result:
            probability=result['probability']
            st.metric('Cancelamento observado',f'{100*probability:.1f}%'.replace('.',','))
            st.write(f'Conclusão observada: **{100*(1-probability):.1f}%**'.replace('.',','))
            st.caption('Probabilidades entre os dois desfechos da amostra. Estudantes ainda em curso não participaram do treinamento.')
            if result['missing']:
                names=', '.join(FEATURE_LABELS.get(k,k) for k in result['missing'])
                st.warning(f'Informações ausentes: {names}. Aplicado o mesmo tratamento do treino: categoria não informada ou mediana.')
            if result['outside_training']:
                names=', '.join(FEATURE_LABELS.get(k,k) for k in result['outside_training'])
                st.warning(f'Fora da faixa observada no treino: {names}. Essa simulação extrapola os valores usados no ajuste.')
        else:
            st.metric('Cancelamento observado','—')
            st.caption('Sem estimativa calculada nesta sessão.')
        st.divider()
        st.write('**Escopo**')
        st.write('Ciências e Tecnologia, UFRN. Perfil e desempenho do primeiro semestre. Modelo XGBoost.')
        st.caption('A falta de datas de cancelamento impede assegurar que o desempenho antecedeu a saída. As características não representam causas individuais de evasão.')
        st.caption('Uso exploratório. Não utilizar para decidir matrícula, bolsas ou acesso a apoio estudantil.')

with methods:
    meta=predictor.metadata
    st.subheader('O que foi avaliado')
    st.write('Cancelamento registrado (classe 1) versus conclusão ou formação (classe 0), entre vínculos com desfecho observado e ao menos uma disciplina avaliada no semestre de ingresso. Não há horizonte de previsão fixo.')
    st.table({'Conjunto':['Treino','Validação','Teste'], 'Coortes':['2014–2017','2018','2019'],
              'Vínculos':[4464,1126,1095], 'Participação':['66,8%','16,8%','16,4%']})
    st.write('Parâmetros escolhidos por Brier em validações temporais internas. Família e limiar escolhidos na validação de 2018. XGBoost empatou com Random Forest no F1-macro dessa validação e teve menor Brier. Não foi reajustado com o teste.')
    st.subheader('Teste retrospectivo · 2019')
    c1,c2,c3=st.columns(3)
    c1.metric('ROC-AUC',f'{meta["test_metrics"]["roc_auc"]:.3f}')
    c2.metric('F1-macro',f'{meta["test_metrics"]["f1_macro"]:.3f}')
    c3.metric('AP',f'{meta["test_metrics"]["ap"]:.3f}')
    st.caption('Referência da AP: 78,0% de cancelamentos no teste. Brier = 0,131. F1-macro e matriz de confusão usam limiar de 0,66, escolhido na validação; a simulação não emite classificação automática.')
    st.table({'Situação observada':['Conclusão','Cancelamento'], 'Predito: conclusão':[144,133], 'Predito: cancelamento':[97,721]})
    st.image(str(ROOT/'assets/comparacao.png'),caption='Mesmos vínculos no teste. Comparação por F1-macro, com limiares escolhidos na validação.')
    with st.expander('Variáveis, pré-processamento e limites'):
        st.write('Sexo, raça/cor, cota, ingresso, escola, idade e renda; número de disciplinas, reprovações, proporção de reprovações, média de notas, média de faltas e proporção de reprovações por faltas. As duas proporções são calculadas a partir das contagens.')
        st.write('Idade aproximada por anos; renda transformada por log(1 + renda). Ausências numéricas recebem a mediana do treino e são padronizadas; categorias usam one-hot. Categorias ausentes não vistas no treino resultam em zeros nesse campo, conforme o pipeline original.')
        st.write('300 árvores, profundidade 2, taxa 0,03, subamostragem de linhas e colunas 0,8, L2 = 1. Probabilidades sem calibração posterior.')
        st.write('O teste já foi consultado em etapas anteriores: esta é uma extensão exploratória. A proporção estimada não foi validada prospectivamente nem para outra instituição. A presença de variáveis sensíveis não autoriza decisões individuais ou inferências causais.')
    st.caption('Pesquisa de Isabele de Lima Nascimento · Ciência da Computação / UNIFESP. Dados de origem: UFRN. Não é um serviço oficial dessas instituições.')
    st.link_button('Código e documentação','https://github.com/isabele-nascimento/bct-predicao',icon=':material/code:')
