import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import scipy.stats as st
import os
import shutil

df = pd.read_csv('/scratch/staff/jrs596/dat/NVAE/eval_HFDS/losses.csv', header=0)



mean = statistics.mean(df['loss'])
sd = statistics.stdev(df['loss'])

#print(mean)
#print(mean+2*sd)

summary = df.describe()

Q1 = summary['loss'][4]
Q3 = summary['loss'][6]
IQR = Q3 - Q1




CI = st.norm.interval(alpha=0.9, loc=np.mean(df['loss']), scale=st.sem(df['loss'])) 
#print('90% CI')
#print(CI)


whis = 2

upper = Q3 + whis*IQR
print(upper)
lower = Q1 - whis*IQR
print(lower)

fig = plt.figure(figsize =(10, 7))
plt.boxplot(df['loss'], showfliers=True, whis=whis)
plt.show()
plt.cla()#

plt.hist(df['loss'])
plt.show()


for i in range(len(df['loss'])):
	if df['loss'][i] > upper:
		source = df['file'][i]
		dest = os.path.join('/scratch/staff/jrs596/dat/NonePlantIM_NVAE', str(i) + '.jpeg')
		shutil.copy(source,dest)