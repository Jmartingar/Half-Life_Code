import pandas as pd

if __name__ == "__main__":
    df_all = pd.read_excel("../raw/Estimating_Serum.xlsx", sheet_name="All", skiprows=1)
    df_unmodified = pd.read_excel("../raw/Estimating_Serum.xlsx", sheet_name="Unmodified", skiprows=2)
    df_modified = pd.read_excel("../raw/Estimating_Serum.xlsx", sheet_name="Modified", skiprows=2)
    
    # Eliminar filas sin secuencia
    df_unmodified = df_unmodified.dropna(subset=["Sequence"])
    df_modified = df_modified.dropna(subset=["Sequence"])
    
    # Revisar si las secuencias están presentes en el dataframe "All"
    df_modified["in_all"] = df_modified["Sequence"].isin(df_all["Sequence"])
    df_unmodified["in_all"] = df_unmodified["Sequence"].isin(df_all["Sequence"])
    
    df_all = df_all[["Sequence", "Half-life (hours)", "Modifications"]].rename(columns={
        "Sequence": "sequence",
        "Modifications": "modifications",
        "Half-life (hours)": "half-life_hours"
    })

    # Agregar columna para filtrar modificaciones
    df_all["is_modified"] = df_all["modifications"].apply(lambda x: True if x != "N.A." else False)
    df_all["half-life"] = pd.to_numeric(df_all["half-life_hours"], errors="coerce").apply(lambda x: x * 3600 if pd.notnull(x) else x)
    df_all = df_all.drop("half-life_hours", axis=1)

    df_duplicados = df_all[df_all.duplicated(subset='sequence')] 
    df_no_duplicados = df_all[~df_all.duplicated(subset='sequence')]

    df_no_duplicados.to_csv("estimating_serum.csv", index=False)