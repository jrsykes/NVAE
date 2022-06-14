import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import scipy.stats as st
import os
import shutil

df = pd.read_csv('/local/scratch/jrs596/dat/NVAE/eval/losses.csv', header=0)

fig = plt.figure(figsize =(10, 7))
plt.boxplot(df['loss'], showfliers=True, whis=1)



mean = statistics.mean(df['loss'])
sd = statistics.stdev(df['loss'])

print(mean)
print(mean+2*sd)

summary = df.describe()
print(summary)
#plt.show()

#plt.hist(df['loss'])


CI = st.norm.interval(alpha=0.9, loc=np.mean(df['loss']), scale=st.sem(df['loss'])) 
print('90% CI')
print(CI)




upper = 105468.789062+(105468.789062-77332.078125)
print(upper)
lower = 77332.078125-(105468.789062-77332.078125)
print(lower)
#plt.show() 
#exit()
for i in range(len(df['loss'])):
	if df['loss'][i] < lower or df['loss'][i] > upper:
		source = df['file'][i]
		dest = os.path.join('/local/scratch/jrs596/dat/NVAE/eval/outliers_combine/test', str(i) + '.jpeg')
		shutil.copy(source,dest)