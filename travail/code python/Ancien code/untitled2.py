import pandas as pd


# =============================================================================
# bos 
# =============================================================================
  
df = pd.read_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/metabodata_bos.csv')
df = df.transpose()
df = df.fillna(0)
df.columns = df.iloc[0,:]
df = df.drop('Unnamed: 0')
table_bos = df.to_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_bos.csv')

# =============================================================================
# cervus
# =============================================================================

df = pd.read_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/metabodata_cervus.csv')
df = df.transpose()
df = df.fillna(0)
df.columns = df.iloc[0,:]
df = df.drop('Unnamed: 0')
table_cervus = df.to_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_cervus.csv')

# =============================================================================
# ovis
# =============================================================================

df = pd.read_csv('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/data_processed_ovis.csv')
df = df.transpose()
df = df.transpose()
df = df.fillna(0)
df = df.set_index([0])
df.columns = df.iloc[0,:]
df = df.drop('Label')
table_ovis = df.to_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_ovis.xlsx')

