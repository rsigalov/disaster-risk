# Loading libraries
import pandas as pd
import glob

# Loading the combined file with all NTM sigmas
df_sigma = pd.read_csv("data/sigma_NTM/sigma_NTM_combined.csv")

# Dealing with dates for faster merging
df_sigma["obs_date"] = pd.to_datetime(df_sigma["obs_date"])
df_sigma["exp_date"] = pd.to_datetime(df_sigma["exp_date"])
df_sigma = df_sigma.rename({"sigma_NTM": "sigma_NTM_new"}, axis = 1) # Changing the name

# Looping through all files in raw_data
svi_filename_list = glob.glob("data/raw_data/svi_params_final_part*")
for (i, filename) in enumerate(svi_filename_list):
	print("File %d out of %d" % (i+1, len(svi_filename_list)))

	# Loading svi parameters file:
	df = pd.read_csv(filename)

	# Removing the old column sigma_NTM_new:
	df = df.drop("sigma_NTM_new", axis = 1)

	# dealing with dates
	df["obs_date"] = pd.to_datetime(df["obs_date"])
	df["exp_date"] = pd.to_datetime(df["exp_date"])

	# Merging and saving
	df = pd.merge(df, df_sigma, on = ["secid", "obs_date", "exp_date"], how = "left")
	df.to_csv(filename, index = False)

	# Printing how many missing value are there in the given dataset:
	print("Total size: %d, missing values: %d" % (df.shape[0], df[df.sigma_NTM_new.isnull()].shape[0]))

