import pandas as pd

Save_PATH = '../his_para/csv/'
DATA_PATH = '../his_para/'
file_name = 'Nearset_WT_1'

# data = pd.read_pickle(DATA_PATH + 'DQN_WT_1' + '.pkl')
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
# pd.set_option('max_colwidth', None)
#
# data = str(data)
# ft = open(EXCEL_PATH + '1' + '.xlsx', 'w')
# ft.write(data)



import pickle
import pandas as pd
f = open(DATA_PATH + file_name + '.pkl', 'rb')
data = pickle.load(f)
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',None)
print(data)
inf=str(data)
ft = open(Save_PATH + file_name + '.csv', 'w')
ft.write(inf)
f.close()
ft.close()
