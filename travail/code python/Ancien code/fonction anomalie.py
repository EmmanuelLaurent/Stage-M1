import numpy as np
import pandas as pd

# =============================================================================
# Importation
# =============================================================================

BDDf = pd.read_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/2-data_ossements/2-2 custom/feature_table_mzwid_0.019_minfrac_0_norm_colors2.xlsx')


# =============================================================================
# fonction
# =============================================================================

def anomalie (os1, os2):
    a=BDDf.loc[BDDf['feature']==os1]
    b=BDDf.loc[BDDf['feature']==os2]
    a2=a.mean(axis=0)
    b2=b.mean(axis=0)
    df = pd.DataFrame([abs(a2-b2)])
    df_pivot = df.T
    return df_pivot.nlargest(20, [0])

# =============================================================================
# teste fonction
# =============================================================================

anomalie('sanglier_crane', 'sanglier_mandibule')
anomalie('humain', 'humain_crane')
    
