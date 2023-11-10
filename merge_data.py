import pandas as pd

# read data
maindata = pd.read_csv('data/Processed_res21_1_clusteredGT.csv')
controls = pd.read_csv('data/international-travel-covid.csv')

# get a map of countries with mismatched names. 
# key is in controls, value in maindata
ref_country_map = {'United States':'USA', 'Czechia':'Czech Republic'}

# rename the column names in controls to match maindata
controls = controls.rename(columns={'Day': 'date', 'Entity':'country'})

# using country map dictionary, replace values to match them with maindata
controls['country'] = controls['country'].replace(ref_country_map)

# there are 185 countries in controls, but only 75 in actual data. 
# remove the ones not present
unique_countries = maindata['country'].unique()
filtered_controls = controls[controls['country'].isin(unique_countries)]

# basically takes a (filtered) data of countries along with a max date value
# then appends additional data from the last available date to the max date using ffill
# this makes sure the prev available value is filled in for any null value
def append_missing_dates(filtered_controls, max_date):
    unique_countries = filtered_controls['country'].unique()
    augmented_controls = []

    for country in unique_countries:

        # get rows for that country 
        country_controls = filtered_controls[filtered_controls['country'] == country]

        # create a date range for country from min to max date
        all_dates = pd.date_range(start=country_controls['date'].min(), end=max_date, freq='D')

        # Create a DataFrame with all_dates and the unique country
        augmented_country_controls = pd.DataFrame({'date': all_dates, 'country': country})

        # make sure dates are in proper format
        country_controls = country_controls.copy()
        country_controls['date'] = pd.to_datetime(country_controls['date'])
        
        # Merge the augmented_country_controls DataFrame with country_controls
        augmented_country_controls = augmented_country_controls.merge(country_controls, on=['country', 'date'], how='left')
        
        # Fill in missing values with the last available values
        augmented_country_controls = augmented_country_controls.fillna(method='ffill')
        
        augmented_controls.append(augmented_country_controls)

    augmented_controls = pd.concat(augmented_controls)
    augmented_controls = augmented_controls.sort_values(by=['country', 'date'])
    
    # do some minor validations before pushing the result
    augmented_controls['international_travel_controls'] = augmented_controls['international_travel_controls'].astype(int)
    augmented_controls['date'] = pd.to_datetime(augmented_controls['date']).dt.strftime('%Y-%m-%d')

    return augmented_controls

# get max value for date
max_date = maindata['date'].max()
augmented_controls = append_missing_dates(filtered_controls, max_date)

# now merge
merged_file = maindata.merge(augmented_controls, on = ['country','date'], how = 'left').drop('Code', axis=1)

# save to csv!
merged_file.to_csv('data/merged_newdata.csv', index=False)