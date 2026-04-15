# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
ds_jobs = pd.read_csv("customer_train.csv")

# View the dataset
ds_jobs.head(10)

# Create a copy of ds_jobs for transforming
dsj = ds_jobs.copy()

# Start coding here. Use as many cells as you like!
dsj.info()

print(dsj.shape)
col_means=[]
col_means=[dsj[column].isna().mean().round(4) for column in dsj.columns ]
map_cmeans= dict(zip(dsj.columns, col_means))
print(map_cmeans)


drop_tcolumns=['gender', 'enrolled_university', 'education_level', 'experience', 'last_new_job']
dsj= dsj.dropna(subset=drop_tcolumns)
print(dsj.shape)

dsj["company_type"]=dsj["company_type"].fillna("Missing")
print(dsj["company_type"].isna().mean())


#automatically inferring that people with no experience are not in a company, hence no company size
dsj.loc[dsj["experience"] == 0, "company_size"] = dsj.loc[dsj["experience"] == 0, "company_size"].fillna(0)
print(dsj["company_size"].isna().mean())

#proportional imputation for the null company size slots
seedd=(81)
np.random.seed(seedd)

probs=dsj["company_size"].value_counts(normalize=True)
csize_mask=dsj["company_size"].isna()

dsj.loc[csize_mask, "company_size"]= np.random.choice( probs.index, size=csize_mask.sum(), p=probs.values)

print(dsj["company_size"].isna().mean())

discp_mode=dsj["major_discipline"].mode()[0]
print(discp_mode)
dsj["major_discipline"]=dsj["major_discipline"].fillna(discp_mode)


print(dsj.shape)
col_means=[]
col_means=[dsj[column].isna().mean().round(4) for column in dsj.columns ]
map_cmeans= dict(zip(dsj.columns, col_means))
print(map_cmeans)

dsj.head(5)
dsj.info()
print(dsj.isna().any())

# Find rows where 'gender' or 'last_new_job' are null
null_rows = dsj[dsj['gender'].isna() | dsj['last_new_job'].isna()]
print(null_rows[['student_id', 'gender', 'last_new_job']])
print(f"Total rows with nulls in 'gender' or 'last_new_job': {null_rows.shape[0]}")

dsj = dsj.dropna(subset=["gender", "last_new_job"])
print(dsj[["gender", "last_new_job"]].isna().any())

#job change seekers as bool
dsj["job_change"]= dsj["job_change"].astype("bool")

dsj["relevant_experience"]=dsj["relevant_experience"].astype(str).str.strip()

#map relevant user experience and gender columns as bool
dsj["relevant_experience"] = dsj["relevant_experience"].map({"Has relevant experience": True, "No relevant experience": False})

#truncate city names to remain with the code, make them integers
dsj["city"]= dsj["city"].replace(r"[^0-9]", "", regex=True)
#trucate unnecessary characters on experience column
#dsj["experience"]=dsj["experience"].replace(r"[^0-9]", '', regex=True)
dsj["experience"]=dsj["experience"].astype(str).str.strip()
#Streamline last new job check column
dsj["last_new_job"]=dsj["last_new_job"].replace({"never": 0 , "Never": 0})
#dsj["last_new_job"]=dsj["last_new_job"].replace(r"[^0-9]", '', regex=True)
dsj["last_new_job"]=dsj["last_new_job"].astype(str).str.strip()

#make all integers  int32
dsj[["student_id", "training_hours" ]]= dsj[["student_id", "training_hours" ]].astype("int32")

#make categorical data categorical
dsj[["city", "gender", "enrolled_university", "education_level", "major_discipline", "company_size", "company_type"]]= dsj[["city", "gender", "enrolled_university", "education_level", "major_discipline", "company_size", "company_type"]].astype("category")

#make all floats float16
dsj["city_development_index"]= dsj["city_development_index"].astype("float16")

#ordered categories
#education level
edu_rank=["Primary School", "High School","Graduate","Masters","Phd"]
dsj["education_level"]= pd.Categorical(dsj["education_level"], categories=edu_rank, ordered=True)
#company size
csize_rank=['<10','10-49','50-99', '100-499','500-999', '1000-4999','5000-9999', '10000+']
dsj["company_size"]=pd.Categorical(dsj["company_size"], categories=csize_rank, ordered=True)
#experience
exp_rank=[str(i) for i in range(1,21)]+[">20"]
dsj["experience"]= pd.Categorical(dsj["experience"], categories=exp_rank, ordered=True)

# last new job
last_job_rank=[str(i) for i in range(1,5)]+[">4"]
dsj["last_new_job"]= pd.Categorical(dsj["last_new_job"], categories=last_job_rank, ordered=True)

#enrolled university
enrol_rank=['no_enrollment', 'Part time course', 'Full time course']
dsj["enrolled_university"]= pd.Categorical(dsj["enrolled_university"], categories=enrol_rank, ordered=True)

dsj.head(5)

mask_xp_csize= (dsj["experience"]>="10") & (dsj["company_size"]>="1000-4999")
dsj= dsj[mask_xp_csize]
dsj.head(10)

ds_jobs_transformed= dsj.sort_values(["education_level", "enrolled_university" , "experience", "company_size", "last_new_job"], ascending=False)
dsj.head(10)
ds_jobs_transformed.info()

ds_jobs_transformed.shape
