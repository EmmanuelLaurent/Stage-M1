# importing the required modules
import glob
import pandas as pd
import os
  
# =============================================================================
# bos
# =============================================================================

chemin = "C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/Bos2"
  
files = glob.glob(chemin + "/*.csv")
  
# defining an empty list to store 
# content
data_frame = pd.DataFrame()
content = []
  
# checking all the csv files in the 
# specified path
for filename in files:
    
    # reading content of csv file
    # content.append(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    df['feature'] = os.path.basename(filename)
    content.append(df)
  
# converting content to data frame
data_frame = pd.concat(content, axis=0, ignore_index=True)
print(data_frame)


data_frame = data_frame.rename(columns={'m/z':"mz"})
data_frame.mz= round(data_frame['mz'], 1)
data_frame = data_frame.drop_duplicates(subset='mz')
data_frame2= data_frame.pivot(index='feature',columns='mz',values='I')
data_frame2 = data_frame2.fillna(0)
table_bovin = data_frame2.to_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_bos.xlsx')

# =============================================================================
# cervus
# =============================================================================

chemin = "C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/cervus2"
  

files = glob.glob(chemin + "/*.csv")
  

data_frame = pd.DataFrame()
content = []
  

for filename in files:
    
    df = pd.read_csv(filename, index_col=None, header=0)
    df['feature'] = os.path.basename(filename)
    content.append(df)
  
# converting content to data frame
data_frame = pd.concat(content, axis=0, ignore_index=True)
print(data_frame)


data_frame = data_frame.rename(columns={'m/z':"mz"})
#data_frame.mz= round(data_frame['mz'], 2)
#data_frame = data_frame.drop_duplicates(subset='mz')
data_frame2= data_frame.pivot(index='feature',columns='mz',values='I')
data_frame2 = data_frame2.fillna(0)
table_bovin = data_frame2.to_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_cervus.xlsx')

# =============================================================================
# ovis
# =============================================================================

chemin = "C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/ovis2"
  
files = glob.glob(chemin + "/*.csv")
  

data_frame = pd.DataFrame()
content = []
  
for filename in files:
    
    df = pd.read_csv(filename, index_col=None, header=0, sep=';')
    df['feature'] = os.path.basename(filename)
    content.append(df)
  
data_frame = pd.concat(content, axis=0, ignore_index=True)
print(data_frame)


data_frame = data_frame.rename(columns={'m/z':"mz"})
#data_frame.mz= round(data_frame['mz'], 2)
#data_frame = data_frame.drop_duplicates(subset='mz')
data_frame2= data_frame.pivot(index='feature',columns='mz',values='I')
data_frame2 = data_frame2.fillna(0)
table_bovin = data_frame2.to_excel('C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/objectif/table_ovis.xlsx')


# =============================================================================
# 
# =============================================================================
BDD = pd.read_excel(
    'C:/Users/orkad/Desktop/stage emmanuel/Datasets-20220517T100241Z-001/Datasets/2-data_ossements/2-2 custom/feature_table_mzwid_0.019_minfrac_0_no_int.xlsx')
