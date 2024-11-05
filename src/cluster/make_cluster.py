# name = "m82"
# name = "high_z"
name = "bursty_5"
# name = "bursty_20"

with open(f'{name}/cluster_list_{name}.txt','r') as ofile:
    data = ofile.read()

data = data.replace('\t',',')
data = data.replace('\n',',\n')
data = data.replace('#','//')
#print(data)

with open(f'{name}/cluster_list_{name}.data','w') as ofile:
    ofile.write('Real cluster_data[] = {\n')
    ofile.write(data)
    ofile.write('};\n')
