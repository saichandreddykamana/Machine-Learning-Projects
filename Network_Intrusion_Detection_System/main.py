import pandas as pd

# Reading data from CSV file
from numpy.ma import count

data = pd.read_csv('data.csv')

# Each Network attack data is saved into a variable
tcp_syn_data = data[data['Label'] == 'TCP-SYN']
normal_data = data[data['Label'] == 'Normal']
blackhole_data = data[data['Label'] == 'Blackhole']
diversion_data = data[data['Label'] == 'Diversion']
overflow_data = data[data['Label'] == 'Overflow']
portscan_data = data[data['Label'] == 'PortScan']

# Converting the list into a dataframe for further uses
tcp_df = pd.DataFrame(tcp_syn_data)
normal_df = pd.DataFrame(normal_data)
blackhole_df = pd.DataFrame(blackhole_data)
diversion_df = pd.DataFrame(diversion_data)
overflow_df = pd.DataFrame(overflow_data)
portscan_df = pd.DataFrame(portscan_data)

