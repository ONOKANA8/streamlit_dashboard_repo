"""
Created on Wed Jul 06 21:30 2022
@author: konan.koffi
"""

import joblib
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def mod_data():
    # chargement du data_test
    path = "fichier_api/fichier-test1000-api.csv"
    data = pd.read_csv(path).drop("Unnamed: 0", axis=1)

    # chargement du modèle entrainé
    path_ = "fichier_api/joblib_lgbm0_Model.pkl"
    model = joblib.load(path_)

    # complétion de data_test avec score, class_bin et class_cat

    score = 100 * model.predict_proba(data.copy(deep=True).iloc[:, :-1])[:, 1]
    class_bin = model.predict(data.iloc[:, :-1])

    class_cat = []
    for i in class_bin:
        if i == 0.0:
            class_cat.append("accepted")
        else:
            class_cat.append("refused")

    data["score"] = score
    data["class_bin"] = class_bin
    data["class_cat"] = class_cat

    # Importation de données d'entrainement pour le tracé de shap
    path__ = "fichier_api/data_tr_api"
    data_tr = pd.read_csv(path__)
    data_tr = data_tr.drop("Unnamed: 0", axis=1)

    return data, data_tr, model


data, data_tr, model = mod_data()

# définition des fonctions utiles


def gauge(ID):
    st.header("Score statistique du client n°{}".format(ID))
    st.subheader("Statut: {}".format(class_cat))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score"},
        gauge={'axis': {'range': [None, 100]},
               'steps': [
                   {'range': [0, 50], 'color': "white"},
                   {'range': [50, 100], 'color': "gray"}],
               'bar': {'color': ["green" if val_score <= 50 else "red"][0]},
               'bgcolor': "white",
               'borderwidth': 2,
               'bordercolor': "gray",
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

    st.plotly_chart(fig)

    st.subheader("Explicabilité du modèle ML")

    # affichage de la feature locale du client via shap
    # construction de l'explicateur shap
    shap_explainer = shap.TreeExplainer(model, data_tr, model_output="probability")
    shap_values = shap_explainer(data.iloc[:, :-4])

    # définition de l'indice en foncion de l'ID
    for idx in data[data["SK_ID_CURR"] == ID].index.values.tolist():
        fig_ = shap.plots.waterfall(shap_values[idx])

    st.pyplot(fig_)


# interactive_plot


def interactive_hist_plot(ID):
    st.subheader("Affichage d'une distribution statistique")
    feature = st.selectbox("Choisir une caratéristique",
                           options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                    "DAYS_EMPLOYED",
                                    "AMT_ANNUITY"])
    fig = px.histogram(data.copy(deep=True), x=feature, color="class_cat", marginal="rug",
                       # can be `box`, `violin`
                       hover_data=data.copy(deep=True).columns, )
    fig.add_trace(go.Histogram(x=data.copy(deep=True)[data.copy(deep=True)["SK_ID_CURR"] == ID][feature],
                               name='client  {}'.format(ID),
                               opacity=1,
                               marker_color='#ffbe0b'))
    fig.update_layout(title_text='Distribution selon {}'.format(feature))

    return st.plotly_chart(fig)


def interactive_scatter_plot(ID):
    st.subheader("Affichage de la relation entre 2 caractéristiques")

    x_axis_val = st.selectbox("Choisir l'abscisse",
                              options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                       "DAYS_EMPLOYED", "AMT_ANNUITY"])
    y_axis_val = st.selectbox("Choisir l'ordonnée",
                              options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                       "DAYS_EMPLOYED", "AMT_ANNUITY"])
    fig = px.scatter(data, x=x_axis_val, y=y_axis_val, color="score", color_continuous_scale='rdbu_r')

    dt = data[data["SK_ID_CURR"] == ID]
    trace = go.Scatter(x=dt[x_axis_val],
                       y=dt[y_axis_val],
                       line_color="yellow", name="client {}".format(ID), visible=True, mode="lines+markers",
                       marker={"size": 20}, fillcolor="white")

    # give it a legend group and hide it from the legend
    trace.update(legendgroup="trendline", showlegend=False)

    # add it to all rows/cols, but not to empty subplots
    fig.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)

    fig.update_layout(autosize=False,
                      width=1000, height=800,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4)
                      )
    return st.write(fig)


st.title("DASHBOARD - VISUALISATION - STATISTIQUE CREDIT")


ID = st.number_input("Saisir l'identifiant du client ici :", key=int, step=1)
if ID not in data.SK_ID_CURR.values:
    st.write("ERREUR : Choisir un identifiant correct")

else:
    r = requests.get('https://pret-a-depenser-heroku.herokuapp.com/credit', params={"ID": ID})
    req = r.json()
    val_score = [j for i, j in req[0].items()][0]
    class_cat = [j for i, j in req[1].items()][0]

    st.set_option('deprecation.showPyplotGlobalUse', False)  # éviter l'affichage d'erreur pyplot


    # action à faire si l'identifiant est rentrer

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Types d'analyse",
                               options=["Score et Explicabilité locale de décision",
                                        "Distribution statistique de caractéristiques",
                                        "Relation entre caractéristiques"])

    if options == "Score et Explicabilité locale de décision":
        gauge(ID)

    elif options == "Distribution statistique de caractéristiques":
        interactive_hist_plot(ID)

    elif options == "Relation entre caractéristiques":
        interactive_scatter_plot(ID)


