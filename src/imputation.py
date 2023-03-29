import numpy as np
import pandas as pd
import pymc3 as pm

# Load the data
full_clinical_data = pd.read_csv("../data/train_clinical_data.csv")
full_clinical_data = pd.concat([full_clinical_data, pd.get_dummies(full_clinical_data['upd23b_clinical_state_on_medication'], prefix='upd23b_medication')], axis=1)
full_clinical_data = full_clinical_data.drop(['upd23b_clinical_state_on_medication'], axis=1)

# Replacing the misisng values in updrs_3 with the previously observed value
predictors = ['updrs_1','updrs_2','updrs_3', 'upd23b_medication_Off', 'upd23b_medication_On']
full_clinical_data[predictors] = full_clinical_data[predictors].fillna(method='bfill')

# Split the data into observed and missing parts
observed_data = full_clinical_data[full_clinical_data['updrs_4'].notnull()]
missing_data = full_clinical_data[full_clinical_data['updrs_4'].isnull()]

# Define the Bayesian regression model
with pm.Model() as model:
    # Define the priors
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    updrs_1_coef = pm.Normal("updrs_1_coef", mu=0, sigma=10)
    updrs_2_coef = pm.Normal("updrs_2_coef", mu=0, sigma=10)
    updrs_3_coef = pm.Normal("updrs_3_coef", mu=0, sigma=10)
    medication_off_coef = pm.Normal("medication_off_coef", mu=0, sigma=10)
    medication_on_coef = pm.Normal("medication_on_coef", mu=0, sigma=10)
    error = pm.HalfCauchy("error", beta=10)

    # Define the linear regression equation
    mu = (
        intercept +
        updrs_1_coef * observed_data["updrs_1"] +
        updrs_2_coef * observed_data["updrs_2"] +
        updrs_3_coef * observed_data["updrs_3"] +
        medication_off_coef * (observed_data["upd23b_medication_Off"] == 1).astype(int) +
        medication_on_coef * (observed_data["upd23b_medication_On"] == 1).astype(int)
    )

    # Define the likelihood
    updrs_4 = pm.Normal("updrs_4", mu=mu, sigma=error, observed=observed_data["updrs_4"])


    # Sample from the posterior distribution
    trace = pm.sample(draws=2000, tune=1000)

# Impute the missing values
imputed_updrs_4 = []
for i, row in missing_data.iterrows():
    imputed_value = (
        trace["intercept"] +
        trace["updrs_1_coef"] * row["updrs_1"] +
        trace["updrs_2_coef"] * row["updrs_2"] +
        trace["updrs_3_coef"] * row["updrs_3"] +
        trace["medication_off_coef"] * (row["upd23b_medication_Off"] == 1) +
        trace["medication_on_coef"] * (row["upd23b_medication_On"] == 1)
    )
    imputed_updrs_4.append(round(max(np.mean(imputed_value),0)))

# Add the imputed values to the original data frame
missing_data["updrs_4"] = imputed_updrs_4
imputed_data = pd.concat([observed_data, missing_data])

imputed_data.to_csv("../data/imputed/train_clinical_imputed.csv")