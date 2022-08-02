"""
Created on Wed Jul 06 21:30 2022
@author: konan.koffi
"""
# ========================================= Importation des librairies ================================================
import joblib
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
import matplotlib.image as mpimg


# ========================================= Importation de données à mettre en cache ==================================


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def mod_data():
    # chargement du data_test
    path = "fichier_api/data_predict.csv"
    data = pd.read_csv(path).drop("Unnamed: 0", axis=1)

    # chargement du modèle entrainé
    path_ = "fichier_api/joblib_lgbm_beta_3_Model.pkl"
    model = joblib.load(path_)

    return data, model


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def shap_val():
    # Importation de données d'entrainement pour le tracé de shap
    path__ = "fichier_api/data_tr_api"
    data_tr = pd.read_csv(path__)
    data_tr = data_tr.drop("Unnamed: 0", axis=1)

    # construction de l'explicateur shap pour les prochaines interpétabilités du modèle
    shap_explainer = shap.TreeExplainer(model, data_tr, model_output="probability")
    shap_values = shap_explainer(data.iloc[:, :-4])

    return shap_values


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def logo():
    path = "fichier_api/logo_pret_a_depenser.png"
    image = mpimg.imread(path)
    return image


# ======================================== Récupération de données ====================================================

data, model = mod_data()
shap_values = shap_val()
image = logo()


# ==================================== Définition des fonctions d'affichage ===========================================

# fonction indiquant le score client à partir d'une jauge
def gauge(ID):
    st.header("Score statistique du client n°{}".format(ID))
    st.subheader("Décision: Demande de crédit {}".format(class_cat))
    st.write("La probabilté de défaut de paiement du client est de ", round(val_score, 2), "%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "SCORE"},
        gauge={'axis': {'range': [None, 100]},
               'steps': [
                   {'range': [0, 50], 'color': "white"},
                   {'range': [50, 100], 'color': "gray"}],
               'bar': {'color': ["green" if val_score <= 50 else "red"][0]},
               'bgcolor': "white",
               'borderwidth': 2,
               'bordercolor': "gray",
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

    return st.plotly_chart(fig), st.write("(*) Ce score représente la probabilité de défaut de paiement")


# Interprétabilité locale du modèle


def local_interpret(ID):
    st.subheader("Interprétabilité locale du modèle")
    max_display = st.slider("Nombre de caractéristiques à afficher", 10, 40, 10, key="w1")
    # définition de l'indice en foncion de l'ID
    for idx in data[data["SK_ID_CURR"] == ID].index.values.tolist():
        fig = shap.plots.waterfall(shap_values[idx], max_display=max_display)

    return st.pyplot(fig)


# Ajout de de l'interprétabilité globale


def global_interpret():
    st.subheader("Interprétabilité globale du modèle")
    max_display = st.slider("Nombre de caractéristiques à afficher", 10, 40, 10, key="w2")
    fig = shap.plots.bar(shap_values, max_display=max_display)

    return st.pyplot(fig)


# interactive_plot, histogramme


def interactive_hist_plot(ID):
    st.subheader("Affichage d'une distribution statistique")
    feature = st.selectbox("Veuillez choisir une caratéristique",
                           options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                    "DAYS_EMPLOYED",
                                    "AMT_ANNUITY"])
    fig = px.histogram(data, x=feature, color="class_cat", marginal="rug",
                       hover_data=data.columns)
    x = data[data["SK_ID_CURR"] == ID]
    fig.add_trace(go.Histogram(x=x[feature],
                               name='client {}'.format(ID),
                               opacity=1,
                               marker_color="#fff000",
                               hovertemplate='<b>%{text}</b>',
                               text=["class_cat : {}, {}={}".format(x.class_cat.values[0],
                                                                    feature, round(x[feature].values[0], 7))
                                     ]
                               )
                  )

    text = "Votre client appartient à cette population"
    fig.add_annotation(
        x=x[feature].values[0],
        y=50,
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#000000"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#e6e64c",
        ax=70,
        ay=-100,
        bordercolor="#000000",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ffffff",
        opacity=1
    )
    fig.update_layout(autosize=False, width=800, height=700,
                      title_text='Distribution selon {}'.format(feature))

    st.plotly_chart(fig)

    impact = st.checkbox("Afficher l'impact de la caractéristique choisie sur la prise de décision du modèle")
    if impact:
        fig_ = px.scatter(data, x=shap_values[:, feature].values, y=feature, color="class_cat",
                          color_continuous_scale='rdbu_r',
                          labels={"x": "Contribution à la probabilité de défaut de paiement"})
        st.plotly_chart(fig_)


# interactive_plot, Nuage de points

def interactive_scatter_plot(ID):
    st.subheader("Affichage de la relation entre 2 caractéristiques")

    x_axis_val = st.selectbox("Veuillez choisir l'abscisse",
                              options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                       "DAYS_EMPLOYED", "AMT_ANNUITY"])
    y_axis_val = st.selectbox("Veuillez choisir l'ordonnée",
                              options=["AMT_INCOME_TOTAL", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
                                       "DAYS_EMPLOYED", "AMT_ANNUITY"])
    fig = px.scatter(data, x=x_axis_val, y=y_axis_val, color="score", color_continuous_scale='rdbu_r')

    dt = data[data["SK_ID_CURR"] == ID]
    trace = go.Scatter(x=dt[x_axis_val],
                       y=dt[y_axis_val],
                       line_color="yellow",
                       name="client {}".format(ID),
                       visible=True, mode="lines+markers",
                       marker={"size": 20}, fillcolor="white",
                       hovertemplate='<b>%{text}</b>',
                       text=["{}={}, {}={}, score={}".format(x_axis_val, round(dt[x_axis_val].values[0], 7),
                                                             y_axis_val, round(dt[y_axis_val].values[0], 7),
                                                             round(val_score, 7))]
                       )

    # give it a legend group and hide it from the legend
    trace.update(legendgroup="trendline", showlegend=False)

    # add it to all rows/cols, but not to empty subplots
    fig.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)

    text = "Votre client est ici"
    fig.add_annotation(
        x=dt[x_axis_val].values[0],
        y=dt[y_axis_val].values[0],
        xref="x",
        yref="y",
        text=text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#000000"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#c40c3a",
        ax=70,
        ay=-100,
        bordercolor="#000000",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ffffff",
        opacity=1
    )
    fig.update_layout(autosize=False,
                      width=800, height=700,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4)
                      )
    return st.write(fig)


# ============================================= Exécution des fonctions d'affichage ===================================

st.write("Dashboard réalisé par Koffi KONAN, Data Scientist chez", "Prêt à dépenser")
st.title("DASHBOARD - VISUALISATION - STATISTIQUE CREDIT")

st.sidebar.image(image, use_column_width=True)

ID = st.number_input("Veuillez saisir un identifiant client ici :", key=int, step=1)
if ID not in data.SK_ID_CURR.values:
    st.write("Veuillez saisir un identifiant client valide pour débuter l'analyse")

else:
    r = requests.get("https://pret-a-depenser-heroku.herokuapp.com/credit", params={"ID": ID})
    req = r.json()
    val_score = [j for i, j in req[0].items()][0]
    class_cat = [j for i, j in req[1].items()][0]

    st.set_option('deprecation.showPyplotGlobalUse', False)  # éviter l'affichage d'erreur pyplot

    # action à faire si l'identifiant est rentré
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Types d'analyse",
                               options=["Score du client et Interprétabilité de décision",
                                        "Distribution statistique de caractéristiques",
                                        "Relation entre caractéristiques"])

    if options == "Score du client et Interprétabilité de décision":

        gauge(ID)
        local_interpret(ID)
        global_ = st.checkbox("Afficher l'interprétabilité globale")
        if global_:
            global_interpret()

    elif options == "Distribution statistique de caractéristiques":
        interactive_hist_plot(ID)

    elif options == "Relation entre caractéristiques":
        interactive_scatter_plot(ID)
