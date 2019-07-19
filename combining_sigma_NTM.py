# Loading libraries
import pandas as pd
import glob

# Getting all files in the folder with sigma_NTM and appending them
sigma_NTM_list = glob.glob("estimated_data/sigma_NTM/*")
for (i, filename) in enumerate(sigma_NTM_list):
	print("File %d out of %d" % (i + 1, len(sigma_NTM_list)))
	if i == 0:
		df_sigma = pd.read_csv(filename)
	else:
		df_sigma = df_sigma.append(pd.read_csv(filename))

df_sigma.to_csv("estimated_data/sigma_NTM/sigma_NTM_combined.csv", index = False)