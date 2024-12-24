import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.compose import make_column_selector


pd.set_option("future.no_silent_downcasting", True)


@st.cache_data
def get_sheet_names(uploaded_file):
    sheet_names = []

    try:
        # Load the Excel file and get  the sheet name
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

    except Exception as e:
        print(f"Error: {e}")

    return sheet_names


# https://docs.streamlit.io/develop/concepts/architecture/caching
def load_pandas_data(uploaded_file, sheet_selected):
    data = pd.DataFrame()

    try:
        data = pd.read_excel(uploaded_file, sheet_name=sheet_selected)

    except Exception as e:
        print(f"Error: {e}")

    return data


@st.cache_data
def analyze_dataframe(df, missing_percent_threshold=1):
    # Set display options for DataFrame
    pd.set_option("display.max_rows", df.shape[0])
    pd.set_option("display.max_columns", df.shape[1])

    # Plot and display the data types
    st.markdown(
        """### Types de données possibles:
    
    * int: nombres entiers
    * bool: valeurs booléennes True=1/ False=0
    * datetime: dates et heures
    * float: nombres décimaux
    * object: texte et colonnes reconnus comme tel avec les fautes de frappes (12,?3 au lieu de 12,3) et annotations (NR, NC, ND...)
    """
    )
    dtype_counts = df.dtypes.value_counts()
    st.bar_chart(dtype_counts)

    st.markdown("### Statistiques par colonne:")

    try:
        # Display statistical details about the DataFrame
        st.markdown("#### Statistiques pour les colonnes numériques:")
        st.table(df.describe(include=[np.number]))
    except Exception:
        st.markdown("Pas de données numériques dans le DataFrame.")

    try:
        st.markdown(
            """
        #### Statistiques pour les colonnes objets:
        Les colonnes objets sont les colonnes avec du texte ou des valeurs catégorielles (couleurs, noms, etc.). Les colonnes numériques reconnues comme objets sont des colonnes avec des fautes de frappe ou des annotations (NR, NC, ND...).
        """
        )
        st.table(df.describe(exclude=[np.number, np.datetime64]))

        unique_bool = df[df.select_dtypes(include=["boolean"]).columns].nunique()

        st.markdown("Colonnes avec une unique valeur (par exemple tout à 0):")
        st.write(unique_bool[unique_bool <= 1])
    except Exception:
        st.markdown("Pas de données catégorielles dans le DataFrame.")

    try:
        st.markdown("#### Statistiques pour les colonnes date:")
        st.table(df.describe(include=[np.datetime64]))
    except Exception:
        st.markdown("Pas de données de type date dans le DataFrame.")

    # Display the number of missing values in each column
    st.markdown("### Valeurs manquantes dans chaque colonne:")
    missing_values = df.isna().sum()
    missing_percent = missing_values / df.shape[0]
    total = df.shape[0]
    col_type = df.dtypes
    missing_stats = pd.DataFrame(
        {
            "Missing Values": missing_values,
            "Total Values": total,
            "Missing Percent": missing_percent,
            "Data Type": col_type,
        },
        index=df.columns,
    )

    missing_stats = missing_stats.sort_values(by="Missing Percent")
    st.table(
        missing_stats[missing_stats["Missing Percent"] < missing_percent_threshold]
    )

    st.markdown(
        """
                ### Visualisation des valeurs manquantes:
    * une ligne blanche = valeur manquante
    * une ligne noire = valeur présente
    """
    )

    fig = px.imshow(
        df.isna(),
        aspect="auto",
        color_continuous_scale=["white", "blue"],
        labels={"color": "Données manquantes"},
    )
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(200, len(df) / 2),  # Dynamic height adjustment
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def regex_column_selector(df, regex_list):
    """like make_column_selector, but using regex list (pattern=r"(?i)date")(df)

    Args:
        df (_type_): _description_
        regex_list (_type_): [ r"var1", r"(?i)VaR2", # (?i) case-insensitive]

    Returns:
        _type_: list of selected columns
    """
    columns_list = []

    for column in df.columns:
        if any(re.search(indicator, column) for indicator in regex_list):
            columns_list.append(column)

    return columns_list


@st.cache_data
def clean_text_abreviation(
    df,
    nan_names=["NR", "ND", "NC", "HR", "°", "Inconnu", "décès\\?\\?\\?"],
    true_names=["ok", "oui"],
    false_names=["non", "no", "refus"],
):
    # construct the insensitive case regex pattern
    nan_pattern = r"(?i)\s*(" + "|".join(nan_names) + r")\s*"
    # Replacement by NaN
    df = df.replace(nan_pattern, np.nan, regex=True)

    true_pattern = r"(?i)\s*(" + "|".join(true_names) + r")\s*"
    df = df.replace(true_pattern, True, regex=True)

    false_pattern = r"(?i)\s*(" + "|".join(false_names) + r")\s*"
    df = df.replace(false_pattern, False, regex=True)

    if "V1 0/1" in df.columns:
        df["V1 0/1"] = df["V1 0/1"].replace(r"^X\s*$", False, regex=True)

    # Remove trailing spaces from each cell
    # df = df.map(lambda x: x.rstrip() if isinstance(x, str) else x)
    df = df.replace(r"^\s+|\s+$", "", regex=True)

    return df


@st.cache_data
def clean_datetime_columns(df):
    # Datetime cleaning
    date_columns = make_column_selector(pattern=r"(?i)date")(df)

    try:
        df[date_columns] = pd.to_datetime(df[date_columns], errors="coerce")
    except Exception:
        print("WARNING: fail to convert datetime")

    # df = df.dropna(subset=date_columns)

    return df


@st.cache_data
def clean_boolean_columns(
    df,
    regex_bool=[r"0/1", r"=1", r"1=", r"≥ 1"],
    bool_cols=[],
):
    # Identify columns to convert to boolean by checking column names for specific patterns

    bool_columns = regex_column_selector(df, regex_bool)

    # Append additional columns directly
    bool_columns.extend(bool_cols)

    # to manage non boolean values
    def convert_to_boolean(value):
        if pd.isna(value):
            return np.nan
        elif value == 0 or value == "0":
            return False
        elif value == 1 or value == "1":
            return True
        else:
            return np.nan

    # Apply the custom function to the identified columns
    df[bool_columns] = df[bool_columns].map(convert_to_boolean)

    # Convert to boolean dtype
    df[bool_columns] = df[bool_columns].astype("boolean")

    return df


@st.cache_data
def clean_numerical_columns(df):
    regex_float = r"(?i)\(L\)|\(%|L/s|\(AA\)|\(m\)|mg/J|\(J\)|c/ml|g\s*/\s*[d]?L|en %$|\(mmHg\)|BIA FFMI"
    columns_float = make_column_selector(pattern=regex_float)(df)

    # Clean float columns
    df = df.replace(r"<1", 0.5, regex=True)
    df[columns_float] = df[columns_float].replace("?", np.nan)
    # in float colomns, clean numbers with double like ,, or ..
    df[columns_float] = df[columns_float].replace(
        r"^(\d+)[.,]{1,2}(\d+)", r"\1.\2", regex=True
    )

    # regex: take the first number, then any non digit character surrounded or not by comma, then the second number
    df[columns_float] = df[columns_float].replace(
        r"^(\d+),{0,1}[^0-9.,],{0,1}(\d+)", r"\1.\2", regex=True
    )

    try:
        df[columns_float] = df[columns_float].replace(
            r"Non réalisable\w*", np.nan, regex=True
        )  # Non réalisable toux incoercible, asthénie
        df[columns_float] = df[columns_float].replace(
            r"Quantité insuffisante\w*", np.nan, regex=True
        )  # Non réalisable toux incoercible, asthénie
        df["Pq G/L"] = df["Pq G/L"].replace(r"(?i)Agreg[ée]e?s?", np.nan, regex=True)

    except Exception:
        pass

    for col in columns_float:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        df["Score GAP"] = df["Score GAP"].astype("Int16")
        df["EA nombre\nsi EA"] = df["EA nombre\nsi EA"].astype("Int16")
        df["Charlson (formule, non ajusté âge)"] = df[
            "Charlson (formule, non ajusté âge)"
        ].astype("Int16")
        df["Dyspnée NYHA (0 à 4)"] = df["Dyspnée NYHA (0 à 4)"].astype("Int16")
    except Exception:
        pass

    return df


@st.cache_data
def clean_data(df, bool_cols=[]):
    """Clean the DataFrame by replacing specific values with np.nan, converting to datetime, and cleaning boolean columns.

    Args:
        df (_type_): _description_
        bool_cols (list, optional): _description_..

    Returns:
        _type_: _description_
    """

    df = clean_text_abreviation(df)
    df = clean_datetime_columns(df)
    df = clean_boolean_columns(df, bool_cols=bool_cols)
    df = clean_numerical_columns(df)

    return df


@st.cache_data
def plot_numerical_data_streamlit(df, target):
    """
    Affiche la distribution des colonnes numériques d'un DataFrame par catégorie dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target (str): La colonne catégorielle par laquelle les données seront séparées.
        standard_deviation (float): Le seuil d'écart-type pour filtrer les valeurs aberrantes.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Supprimer target des colonnes numériques si elle est présente
    numeric_columns = [col for col in numeric_columns if col != target]

    for col in numeric_columns:
        st.subheader(f"Distribution de {col} par {target}")

        fig, ax1 = plt.subplots(figsize=(8, 4))  # Nouveau graphique pour chaque colonne

        # Second axe pour les valeurs réelles
        ax2 = ax1.twinx()

        # Boucle sur chaque catégorie dans la colonne spécifiée
        for category in df[target].unique():
            category_df = df[df[target] == category]

            # Tracer la distribution (pourcentage) sur l'axe de gauche
            sns.histplot(
                category_df[col],
                label=f"{category}",
                kde=True,
                stat="percent",
                common_norm=True,
                alpha=0.25,
                ax=ax1,
            )

            # Tracer les fréquences absolues sur l'axe de droite
            sns.histplot(
                category_df[col],
                kde=False,
                stat="count",
                alpha=0.1,
                ax=ax2,
            )

        # Ajouter des titres et légendes
        ax1.set_title(f"Distribution of {col} by {target}")
        ax1.set_ylabel("Percentage")
        ax2.set_ylabel("Count")
        ax1.legend(title=target)
        plt.tight_layout()

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)


@st.cache_data
def plot_crosstab_streamlit(df, target):
    """
    Affiche des heatmaps des tableaux croisés dynamiques (crosstab) pour les colonnes non numériques et non temporelles
    d'un DataFrame par rapport à une colonne cible dans Streamlit.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target (str): La colonne cible utilisée pour le tableau croisé.
    """
    non_numeric_columns = df.select_dtypes(exclude=["number", "datetime"]).columns

    for col in non_numeric_columns:
        st.subheader(f"Tableau dynamique croisé de {col} par {target}")

        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            heatmap_data = pd.crosstab(df[target], df[col], dropna=False)
            sns.heatmap(heatmap_data, annot=True, fmt="d", ax=ax)

            ax.set_title(f"{target} vs {col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Afficher dans Streamlit
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur avec {col}: {e}")


@st.cache_data
def test_y_quali_X_quanti(df: pd.DataFrame, _test_stat, target_col: str, alpha=0.05):
    """Calculate the T-test for the means of two independent samples of scores.
    #https://www.bibmath.net/dico/index.php?action=affiche&quoi=./s/studenttest.html
    #https://docs.scipy.org/doc/scipy/reference/stats.html

    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

    Args:
        df (pd.DataFrame): _description_
        target_col (str): _description_
        alpha (float, optional): pvalue

    Returns:
        _type_: _description_
    """
    # Séparation des données en fonction de la colonne cible
    positive_df = df[df[target_col]]  # True
    negative_df = df[~df[target_col]]  # False

    # Échantillonnage équilibré du groupe négatif pour avoir la même taille que le groupe positif
    balanced_neg = negative_df.sample(positive_df.shape[0])

    # DataFrame pour stocker les résultats
    df_result = pd.DataFrame()
    n_nan = df.isna().sum() / df.shape[0] * 100
    # df_result['nan'] = n_nan

    # Parcourir chaque colonne du DataFrame (sauf la colonne cible)
    for col in df.select_dtypes(include=["number"]).columns:
        if col != target_col:  # On ignore la colonne cible
            # Effectuer le test t pour chaque colonne
            # stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
            (
                statistic,
                pvalue,
            ) = _test_stat(balanced_neg[col], positive_df[col], nan_policy="omit")

            # Vérifier si l'hypothèse nulle est rejetée
            result_str = (
                f"H0 Rejetée avec un risque d'erreur <{alpha*100}%"
                if pvalue < alpha
                else "H0 Accepté, moyennes égales"
            )

            # Création d'un DataFrame temporaire pour la colonne actuelle
            df_current = pd.DataFrame(
                {
                    "missing": [n_nan[col]],
                    "stat": [statistic],
                    "pvalue": [pvalue],
                    "result": [result_str],
                },
                index=[col],
            )

            # Concatenation des résultats pour chaque colonne
            df_result = pd.concat([df_result, df_current])

    # Format the 'missing %' column as a percentage
    df_result["missing"] = df_result["missing"].map(lambda x: f"{x:.2f}%")

    # Retourner les résultats triés par la p-valeur
    return df_result.sort_values(by="pvalue")


@st.cache_data
def test_y_quali_X_quali(df: pd.DataFrame, _test_stat, target_col: str, alpha=0.05):
    """Calculate the T-test for the means of two independent samples of scores.
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

    Args:
        df (pd.DataFrame): _description_
        target_col (str): _description_
        alpha (float, optional): pvalue

    Returns:
        _type_: _description_
    """

    n_nan = df.isna().sum() / df.shape[0] * 100

    # DataFrame pour stocker les résultats
    df_result = pd.DataFrame()

    # Parcourir chaque colonne du DataFrame (sauf la colonne cible)
    for col in df.columns:
        if col != target_col:  # On ignore la colonne cible
            obs = pd.crosstab(df[target_col], df[col])
            # Effectuer le test pour chaque colonne
            statistic, pvalue, dof, expected_freq = _test_stat(obs, correction=False)

            # Vérifier si l'hypothèse nulle est rejetée
            result_str = (
                f"H0 Rejetée avec un risque d'erreur <{alpha*100}%"
                if pvalue < alpha
                else "H0 Accepté, variables indépendantes"
            )

            # Création d'un DataFrame temporaire pour la colonne actuelle
            df_current = pd.DataFrame(
                {
                    "missing": [n_nan[col]],
                    "stat": [statistic],
                    "pvalue": [pvalue],
                    "result": [result_str],
                },
                index=[col],
            )

            # Concatenation des résultats pour chaque colonne
            df_result = pd.concat([df_result, df_current])

    # Format the 'missing %' column as a percentage
    df_result["missing"] = df_result["missing"].map(lambda x: f"{x:.2f}%")

    # Retourner les résultats triés par la p-valeur
    return df_result.sort_values(by="pvalue")


def accueil():
    st.markdown(
        """
        Cette application facilite l'analyse de données médicales au format Excel, avec un focus sur les exacerbations liées à la Fibrose Pulmonaire Idiopathique (FPI). Plusieurs types d'analyses sont disponibles :
        ## 📊 Analyse de la forme des données

        * Identification des types de données (numérique, catégoriel, etc.)
        * Détection des valeurs manquantes et anomalies
        * Résumé statistique des variables

        ## 📈 Analyse de distribution et croisé
        
        Analyse en fonction d'une variable cible catégorielle (exacerbation FPI oui ou non)
        
        ## 🎨 Visualisation libre

        Un espace interactif pour créer vos propres graphiques
        """
    )


@st.cache_resource
def predict_fpi(df):
    # Load the pre-trained model
    model_file = "fpi.joblib"
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        raise RuntimeError(
            f"Model file {model_file} not found. Please ensure the model file exists in the current directory."
        )

    # Predict the target variable
    y_pred_proba = model.predict_proba(df)

    return y_pred_proba


def prediction_window() -> None:
    selected_features = [
        "BAI",
        "Charlson (formule, non ajusté âge)",
        "IDM 0/1",
        "Taille (m)",
        "Dyspnée NYHA (0 à 4)",
        "PA (uniquement si tabac)",
    ]

    st.header("Saisie des données")

    # Initialisation du DataFrame éditable avec colonnes numériques par défaut
    df_editable = pd.DataFrame(
        data={col: pd.Series(dtype=float) for col in selected_features},
        index=range(1),
    )

    # Configuration des colonnes en fonction de leurs types
    column_config = {
        "BAI": st.column_config.NumberColumn(
            "BAI",
            format="%.2f",
            min_value=0,
            max_value=100,  # Ajustez selon la plage attendue
        ),
        "Charlson (formule, non ajusté âge)": st.column_config.NumberColumn(
            "Charlson (formule, non ajusté âge)",
            format="%.2f",
            min_value=0,
            help="Indice de comorbidité de Charlson non ajusté à l'âge.",
        ),
        "IDM 0/1": st.column_config.NumberColumn(
            "IDM 0/1",
            format="%.0f",
            min_value=0,
            max_value=1,  # Valeurs binaires
            help="Indicateur binaire : 0 pour absence, 1 pour présence d'infarctus du myocarde.",
        ),
        "Taille (m)": st.column_config.NumberColumn(
            "Taille (m)",
            format="%.2f",
            min_value=1.0,
            max_value=2.5,  # Taille humaine réaliste
            help="Taille en mètres (valeurs entre 1.0 et 2.5).",
        ),
        "Dyspnée NYHA (0 à 4)": st.column_config.NumberColumn(
            "Dyspnée NYHA (0 à 4)",
            format="%.0f",
            min_value=0,
            max_value=4,  # Scores de 0 à 4
            help="Score NYHA (0 : pas de dyspnée, 4 : dyspnée sévère).",
        ),
        "PA (uniquement si tabac)": st.column_config.NumberColumn(
            "PA (uniquement si tabac)",
            format="%.2f",
            min_value=0,
            max_value=100,  # Ajustez selon les données attendues
            help="Nombre de paquets-années, uniquement si le patient est fumeur.",
        ),
    }

    # Éditeur Streamlit avec configuration des colonnes
    edited_df = st.data_editor(
        df_editable,
        # num_rows="dynamic",
        column_config=column_config,
        use_container_width=True,
    )

    st.markdown(
        """
    ## 🤖 Prédictions des exacerbations FPI
    """
    )
    if st.button("Prédire", key="predict"):
        # Vérifier si les données sont valides (sans NaN)
        if edited_df.isnull().values.all():
            st.error("Veuillez remplir les colonnes avant de prédire.")
        else:
            try:
                df_pred = predict_fpi(edited_df)
                st.success("Prédictions effectuées avec succès !")

                #for i, prob in enumerate(df_pred[:, 1]):
                #    st.write(
                #        f"Patient {i + 1}: **{prob * 100:.2f}%** probabilité d'exacerbation FPI"
                #    )
                st.write(
                    f"##### Probabilité **{df_pred[0, 1] * 100:.2f}%** d'exacerbation FPI"
                )

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la prédiction : {e}")

    st.markdown("""
                ### Explicabilité du modèle
                
                Ce modéle est basé sur un ensemble d'arbres de décision [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html). Il est capable de prédire les exacerbations de la Fibrose Pulmonaire Idiopathique (FPI) en fonction de plusieurs variables médicales.
                
                - **Importance des variables** : Le score de dyspnée NYHA (essoufflement) et l'indice de comorbidité de Charlson jouent un rôle déterminant dans la prédiction des exacerbations. Par conséquent, ces deux indices déjà utilisés sont pertinents pour anticiper les exacerbations. Ce modèle affine légèrement la prédiction.
                
                La figure illustre la contribution des variables du modèle pour ajuster la valeur de base (moyenne calculée sur l'ensemble du jeu de données d'entraînement) vers la valeur prédite pour un exemple donné. Les variables qui augmentent la prédiction sont représentées en rouge tandis que celles qui la diminuent sont en bleu ([lien article](https://www.nature.com/articles/s42256-019-0138-9.epdf) ).
                """
    )
    try:
        st.image(
            "assets/exacerbations/features_importances.png",
            caption="Importance des features",
        )
        
    except Exception as e:
        #st.error(f"Erreur lors de l'affichage de l'image : {e}")
        st.image(
        "https://adrien-gauche.github.io/portfolio/assets/exacerbations/features_importances.png",
        caption="Importance des features",
        )

    st.markdown(
        """
                - **Seuil de précision** : Le seuil de précision à 50% est correct pour éviter trop de faux positifs et négatifs.
                
                Une précision élevée est obtenue avec peu de faux positifs dans les résultats prédits, et un rappel (recall) élevé est obtenu en ayant peu de faux négatifs [explication sklearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).
                """
    )
    try:
        st.image(
            "assets/exacerbations/precision_threshold.png", caption="Seuil de précision"
        )
    except Exception as e:
        #st.error(f"Erreur lors de l'affichage de l'image : {e}")
        st.image(
            "https://adrien-gauche.github.io/portfolio/assets/exacerbations/precision_threshold.png",
            caption="Importance des features",
        )

    st.markdown(
        """
                - **Courbe Receiver Operating Characteristic (ROC)** : Le modèle prédit correctement les vrais positifs et négatifs.
                
                La courbe ROC, est un graphique qui illustre les performances d'un système de classification binaire lorsque son seuil de discrimination varie. Elle est créée en traçant la fraction des vrais positifs parmi les positifs (TPR = taux de vrais positifs) par rapport à la fraction des faux positifs parmi les négatifs (FPR = taux de faux positifs), à différents seuils [explication ROC](https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc)
                """
    )
    try:
        st.image("assets/exacerbations/ROC.png", caption="Courbe ROC")
    except Exception as e:
        #st.error(f"Erreur lors de l'affichage de l'image : {e}")
        st.image(
            "https://adrien-gauche.github.io/portfolio/assets/exacerbations/ROC.png",
            caption="Importance des features",
        )
