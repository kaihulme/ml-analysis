# reducing complexity

# corr for features
print(corr["MedHouseVal"].abs().sort_values(ascending=False))

# drop features with low corr
drop_features = ['AveBedrmsPerOccup', 'EstHouses', 'AveBedrms',
                 'Population', 'AveOccup']
X_train_less = X_train.drop(drop_features, axis=1, inplace=False)
data_train_less = data_train.drop(drop_features, axis=1, inplace=False)

# optimisation

# columns and feature histogram
data_train_less.columns
data_train_less.hist(figsize=(12, 8))

# lognorm colunmn and new hist
new_data = data_train_less

new_data["MedHouseVal"] = np.log1p(new_data['MedHouseVal'])
new_data["MedInc"] = np.log1p(new_data['MedInc'])
new_data["AveRooms"] = np.log1p(new_data['AveRooms'])
new_data["AveBedrmsPerRoom"] = np.log1p(new_data['AveBedrmsPerRoom'])
new_data["AveAddRooms"] = np.log1p(new_data['AveAddRooms'])

new_data.hist(figsize=(12, 8))

# LA location as north-westerly peak
la_lat, la_lon = 33, -117
# SF location as south-easterly peak
sf_lat, sf_lon = 37, -121

new_data['DistToLA'] = np.sqrt((new_data['Latitude'] - la_lat)**2 +
                               (new_data['Longitude'] - la_lon)**2)
new_data['DistToSF'] = np.sqrt((new_data['Latitude'] - sf_lat)**2 +
                               (new_data['Longitude'] - sf_lon)**2)

new_data['DistToCity'] = new_data[['DistToLA', 'DistToSF']].min(axis=1)

# get new featuers
features = ['MedHouseVal', 'MedInc', 'HouseAge', 'AveBedrmsPerRoom',
            'AveAddRooms', 'DistToCity']

new_data = new_data[features]

new_data.hist(figsize=(12, 8))

# scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)

print(scaled_data)

scaled_df = pd.DataFrame(scaled_data, columns=features)

scaled_df.hist(figsize=(12, 8), bins=50)

# nul check
scaled_df.isnull().sum()

#### model ####

formula = "MedHouseVal ~ MedInc + HouseAge +\
           AveBedrmsPerRoom + AveAddRooms + DistToCity"

with pm.Model() as model:
    # Normal prior
    family = pm.glm.families.Normal()
    # Create model from formula
    pm.GLM.from_formula(formula, data=scaled_df, family=family)
    trace = pm.sample(draws=5000, tune=2000,
                      progressbar=True, cores=16)