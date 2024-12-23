import streamlit as st
from functions import *
from pygwalker.api.streamlit import StreamlitRenderer

st.set_page_config(page_title="Analyses des données FPI", page_icon="📈", layout="wide")

# https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory

options_vues = {
    "accueil": "Accueil",
    "forme": "Analyse de forme des données",
    "distribution": "Distribution variables continues",
    "repartition": "Répartition variables catégorielles",
    "libres": "Analyses libres",
    "survie": "Courbes de survie",
    "pred": "Prédictions FPI",
}


st.sidebar.markdown("# Chargement des données")

uploaded_file = st.sidebar.file_uploader(
    "Choisissez un fichier Excel (.xlsx uniquement)", type=["xlsx"]
)

if uploaded_file is not None:
    # df = get_sheet_names(uploaded_file)
    sheet_names = get_sheet_names(uploaded_file)

    if sheet_names:
        sheet_selected = st.sidebar.selectbox(
            "Sélectionnez l'onglet Excel",
            options=sheet_names,  # Utilisez les noms de feuilles obtenus
            index=1,  # Par défaut, sélectionnez la première feuille
        )

        # Chargez les données de la feuille sélectionnée
        data = load_pandas_data(uploaded_file, sheet_selected)

        # Option de nettoyage des données
        r_clean_data = st.sidebar.checkbox(
            "Nettoyer les données (fautes de frappe, suppression du texte...)",
            value=False,
        )

        if r_clean_data:
            df = clean_data(
                data, bool_cols=["BAI", "IgG4", "Expositions PRO", "Exposition EVT"]
            )
        else:
            df = data

        # Sélectionnez le mode d'analyse
        sb_mode = st.sidebar.selectbox(
            "Choisissez le mode d'analyse",
            options=list(options_vues.values()),  # Utilisez les valeurs de options_vues
            index=0,
        )

        # Exécutez l'affichage en fonction du mode sélectionné

        if sb_mode == options_vues["accueil"]:
            st.title(
                f"Accueil Application d'analyses et de prédictions des exacerbations FPI"
            )

            accueil()

        if sb_mode == options_vues["forme"]:
            st.title(options_vues["forme"])
            st.markdown(
                """ Analysez les données des exacerbations de la Fibrose Pulmonaire Idiopathique (FPI)
                """
            )
            analyze_dataframe(df)
            
        elif sb_mode == options_vues["distribution"]:
            st.title(options_vues["forme"])

            selection = df.select_dtypes(exclude=["number", "datetime"]).columns
            sb_var_to_plot = st.selectbox(
                "Variable catégorielle à visualiser (maladie 0/1...)",
                selection,
            )

            if sb_var_to_plot is not None:
                st.header(
                    f"Variables numériques suivant la variable catégorielle {sb_var_to_plot}"
                )

                st.write("## Test de Student")
                sl_pvalue_quanti = st.slider("pvalue_quanti", 0.01, 0.1, 0.05)

                df_y_quali_X_quanti = test_y_quali_X_quanti(
                    df,
                    stats.ttest_ind,
                    target_col="EA FPI\n>ou= à 1 EA 0/1",
                    alpha=sl_pvalue_quanti,
                )
                st.write(df_y_quali_X_quanti)

                plot_numerical_data_streamlit(df, sb_var_to_plot)
                
        elif sb_mode == options_vues["repartition"]:
            st.title(options_vues["forme"])

            selection = df.select_dtypes(exclude=["number", "datetime"]).columns
            sb_var_to_plot = st.selectbox(
                "Variable catégorielle à visualiser (maladie 0/1...)",
                selection,
            )

            if sb_var_to_plot is not None:

                st.header(
                    f"Variables catégorielles suivant la variable catégorielle {sb_var_to_plot}"
                )

                st.write("Test du Chi2")
                sl_pvalue_quali = st.slider("pvalue_quali", 0.01, 0.1, 0.05)

                df_y_quali_X_quali = test_y_quali_X_quali(
                    df.select_dtypes(include=["boolean"]),
                    stats.chi2_contingency,
                    target_col="EA FPI\n>ou= à 1 EA 0/1",
                    alpha=sl_pvalue_quali,
                )
                st.write(df_y_quali_X_quali)

                plot_crosstab_streamlit(df, sb_var_to_plot)

        elif sb_mode == options_vues["libres"]:
            st.title(options_vues["libres"])

            st.markdown(
                """
            Permet d'afficher librement une variable en fonction d'une autre. La documentation de cet outil est ici: [PygWalker](https://docs.kanaries.net/fr/pygwalker).
            """
            )
            pyg_app = StreamlitRenderer(df.select_dtypes(exclude=["object"]))
            pyg_app.explorer()

        elif sb_mode == options_vues["survie"]:
            st.title(options_vues["survie"])

        elif sb_mode == options_vues["pred"]:
            st.title(options_vues["pred"])

            selected_features = [
                "BAI",
                "Charlson (formule, non ajusté âge)",
                "IDM 0/1",
                "Taille (m)",
                "Dyspnée NYHA (0 à 4)",
                "PA (uniquement si tabac)",
            ]

            st.header("Saisie des données")

            # Nombre de lignes dynamiques
            num_rows = st.number_input(
                "Nombre de patients", min_value=1, value=1, step=1
            )

            # Initialisation du DataFrame éditable avec colonnes numériques par défaut
            df_editable = pd.DataFrame(
                data={col: pd.Series(dtype=float) for col in selected_features},
                index=range(num_rows),
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
                ),
                "IDM 0/1": st.column_config.NumberColumn(
                    "IDM 0/1",
                    format="%.0f",
                    min_value=0,
                    max_value=1,  # Valeurs binaires
                ),
                "Taille (m)": st.column_config.NumberColumn(
                    "Taille (m)",
                    format="%.2f",
                    min_value=1.0,
                    max_value=2.5,  # Taille humaine réaliste
                ),
                "Dyspnée NYHA (0 à 4)": st.column_config.NumberColumn(
                    "Dyspnée NYHA (0 à 4)",
                    format="%.0f",
                    min_value=0,
                    max_value=4,  # Scores de 0 à 4
                ),
                "PA (uniquement si tabac)": st.column_config.NumberColumn(
                    "PA (uniquement si tabac)",
                    format="%.2f",
                    min_value=0,
                    max_value=100,  # Ajustez selon les données attendues
                ),
            }

            # Éditeur Streamlit avec configuration des colonnes
            edited_df = st.data_editor(
                df_editable,
                num_rows="dynamic",
                column_config=column_config,
            )

            st.header("Prédictions du risque d'exacerbations FPI")
            if st.button("Prédire", key="predict"):
                # Vérifier si les données sont valides (sans NaN)
                if edited_df.isnull().values.any():
                    st.error("Veuillez remplir toutes les colonnes avant de prédire.")
                else:
                    # Appel de la fonction de prédiction
                    edited_df = edited_df.dropna()
                    print(edited_df)
                    df_pred = predict_fpi(edited_df)
                    st.write("Résultats des prédictions :", df_pred)

    else:
        st.error("Le fichier Excel n'a pas de feuilles ou est corrompu.")

else:
    st.title(f"Accueil Application d'analyses et de prédictions des exacerbations FPI")

    st.info("Veuillez charger un fichier Excel à gauche pour continuer.")

    accueil()
