# BCT / UFRN

Demonstração de pesquisa com perfil de ingresso e desempenho do primeiro
semestre do curso de Ciências e Tecnologia da UFRN.

**Aplicativo:** https://bct-predicao-ufrn.streamlit.app/

## O que o resultado significa

O XGBoost estima cancelamento (1) versus conclusão/formação (0) **entre os
desfechos observados na amostra**. Estudantes ainda em curso não participaram
do treinamento. Não é uma estimativa validada de evasão futura nem há prazo
fixo de previsão. Não usar para decisões individuais sobre matrícula, bolsas
ou acesso a apoio estudantil.

## Modelo e avaliação

| Conjunto | Coortes | Vínculos |
|---|---|---:|
| Treino | 2014–2017 | 4.464 |
| Validação | 2018 | 1.126 |
| Teste | 2019 | 1.095 |

Parâmetros escolhidos pelo Brier em três validações temporais internas ao
treino. Família e limiar selecionados na validação final. XGBoost e Random
Forest empataram em F1-macro nessa validação; XGBoost teve menor Brier.
O modelo não foi reajustado com o teste.

No teste: ROC-AUC **0,823**, AP **0,942**, F1-macro **0,709**, Brier **0,131**.
A prevalência de cancelamento era **78,0%**, referência para interpretar AP.
No limiar 0,66: TP=721, FP=97, FN=133, TN=144. A página mostra a probabilidade,
sem emitir uma decisão automática pelo limiar.

A falta de datas de cancelamento impede garantir que os indicadores
acadêmicos antecederam a saída. O teste já foi consultado em etapas anteriores
do TCC: esta é uma extensão exploratória, não uma validação independente.

## Entradas e preparo

- Perfil: sexo, raça/cor, cota, forma de ingresso, escola, idade e renda.
- Desempenho: disciplinas avaliadas, reprovações, proporção de reprovações,
  média das notas, média das faltas e proporção de reprovações por faltas.
- Somente componentes aprovados/reprovados do semestre de ingresso; médias
  aritméticas, sem ponderação por créditos. Média de notas não é CR.
- Idade: ano de ingresso menos ano de nascimento. Renda: `log(1 + renda)`.
  O dicionário diz “renda do discente no período”, sem periodicidade explícita;
  não foi convertida em renda familiar, per capita ou salários mínimos.
- Numéricas ausentes: mediana do treino e padronização. Categóricas: categoria
  ausente e one-hot; categorias não vistas resultam em zeros naquele campo.
  As proporções são derivadas das contagens para evitar entradas contraditórias.

## Executar e testar

Python **3.11 ou 3.12**, em ambiente separado:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m unittest discover -s tests -v
```

No Streamlit Community Cloud: repositório `isabele-nascimento/bct-predicao`,
branch `main`, arquivo `app.py`. Não são necessários segredos ou conexão a banco.

## Arquivos e verificação

`app.py` contém a interface; `inference.py`, a inferência. `model/` contém as
árvores em JSON, parâmetros do pré-processamento, métricas e versões de treino.
Não há pickle nem dependência do scikit-learn na inferência. O arquivo
`requirements.txt` registra as versões de execução; o pandas da interface pode
diferir do usado no treino porque o preparo exportado é executado em NumPy.

A exportação reproduziu exatamente o preparo e as probabilidades do notebook
nos **6.685 vínculos**, sem exportá-los. Os casos de teste do repositório são
fictícios. Os testes cobrem paridade, entradas inválidas, limpeza da sessão,
integridade dos artefatos e quebras de linha portáveis entre Windows e Linux.

## Dados e privacidade

Não contém bases individuais, nomes, matrículas ou CPFs. O aplicativo não grava
os perfis; os campos permanecem na sessão e são processados no servidor da
hospedagem. Não há banco, formulário de cadastro nem telemetria própria; isso
não elimina os registros técnicos que a plataforma de hospedagem possa manter.
O modelo e as estatísticas aprendidas são públicos, sem garantia formal de
privacidade diferencial. Use preferencialmente perfis fictícios.

Pesquisa de **Isabele de Lima Nascimento**, Ciência da Computação / UNIFESP.
Dados de origem: UFRN. Não é um serviço oficial dessas instituições.
