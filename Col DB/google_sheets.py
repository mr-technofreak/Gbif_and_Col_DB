# google_sheets.py

import pandas as pd
import gspread
from google.oauth2 import service_account
import os
import logging
import re
from geotext import GeoText
import spacy
import geonamescache
import csv
import time



# Set a format to the logs.
LOG_FORMAT = '[%(levelname)s  | %(asctime)s] - %(message)s'

# Name of the file to store the logs.
LOG_FILENAME = 'script_execution.log'

# Level in which messages are to be logged. 
LOG_LEVEL = logging.DEBUG

# == Set up logging ============================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"),
              logging.StreamHandler()]
)


# == Script Start ==============================================================
# Log the script execution start
logging.info('Script started execution!')

# Load the SpaCy model
nlp = spacy.load("/Users/sairamakshayakumar/GBIF_and_ColDp_Archives/PROJECT/en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0")

# Load the geonamescache country data
gc = geonamescache.GeonamesCache()
countries = gc.get_countries_by_names()

# specify data and project folders
data_folder = "/Users/sairamakshayakumar/GBIF_and_ColDp_Archives/Archives/Catalogue_of_life/DwcA"
output_folder =  "/Users/sairamakshayakumar/GBIF_and_ColDp_Archives/PROJECT"

# Load the credentials and authenticate
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = service_account.Credentials.from_service_account_file("credentials.json", scopes=scopes)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet_id = "1nUHVUJpmMffuB-sQ_caVUAD6gpK2KSnGVdOTLw_Ztl0"
sheet_name = "Names_Src_Details"
sheet = client.open_by_key(sheet_id).worksheet(sheet_name)

# Read the data into a pandas DataFrame
try:
    data = sheet.get_all_values()
    headers = data[2]  # Use only the 3rd row as column names
    rows = data[4:]  # Actual data starts from the 5th row
    df = pd.DataFrame(rows, columns=headers)
    df = df.iloc[:600,:40]  # Keep only the first 40 columns
    logging.info("Successfully read data from Google Sheet.")
except Exception as e:
    logging.error(f"Error while reading data from Google Sheet: {str(e)}")
    # Ensure unique indices: Before assigning the index to 'Original Order', make sure the index of the DataFrame is unique.
df.reset_index(inplace=True, drop=True)
# Add a new column to store the original order of the rows
df['Original Order'] = df.index
logging.info("Added a new column to store the original order of the rows.") 
used_row_orders = set()  # Keep track of used row orders

# After creating 'df', write it to a CSV file
try:
    df.to_csv('google_sheets_data.csv', index=False)
    logging.info("Successfully wrote sheets data to google_sheets_data.csv file.")
except Exception as e:
    logging.error(f"Error while writing sheets data to google_sheets_data.csv file: {str(e)}")  


# Read Taxon.tsv
taxon_file = os.path.join(data_folder, "Taxon.tsv")
try:
    taxon_df = pd.read_csv(taxon_file, sep='\t', quoting=3, quotechar='"', on_bad_lines='skip', low_memory=False)
    logging.info("Successfully read Taxon.tsv.")
except Exception as e:
    logging.error(f"Error while reading Taxon.tsv: {str(e)}")

# Read Distribution.tsv
distribution_file = os.path.join(data_folder, "Distribution.tsv")
try:
    distribution_df = pd.read_csv(distribution_file, sep='\t', quoting=3, quotechar='"', low_memory=False)
    logging.info("Successfully read Distribution.tsv.")
except Exception as e:
    logging.error(f"Error while reading Distribution.tsv: {str(e)}")

# Load valid countries from CSV file
#try:
    ## Load the list of valid countries
    #valid_countries_df = pd.read_csv('country_list.csv')
    #valid_countries = set(valid_countries_df['country'].str.lower())
    #logging.info("Successfully loaded valid countries.")    
#except Exception as e:
    #logging.error(f"Error while reading country_list.csv: {str(e)}")

# Clean and preprocess names

# strip_authorship
def strip_authorship(name):
    return name.split(',')[0]

# clean name
def clean_name(name):
    name = re.sub(r'[*.]', '', name)  # Remove special characters
    name = re.sub(r'\s+', ' ', name)  # Ensure there's only a single space between words
    return name.strip().lower()

# Clean and preprocess the species names in the Google Sheet DataFrame
df['Cleaned Name'] = df['Bi or Trinomial Name'].apply(clean_name)
logging.info("Cleaned names in Google Sheets DataFrame.")

# Clean and preprocess the species names in the Taxon DataFrame
taxon_df['Cleaned Scientific Name'] = taxon_df['dwc:scientificName'].apply(strip_authorship).apply(clean_name)
logging.info("Cleaned names in Taxon DataFrame.")



# Search Function

def split_species_name(name):
    """
    This function splits a name into genus, species, and subspecies
    """
    parts = name.split(' ')
    genus = parts[0]
    species = parts[1] if len(parts) > 1 else None
    subspecies = parts[2] if len(parts) > 2 else None

    return genus, species, subspecies

def search_species(row):
    try:
        name = row['Cleaned Name']
        # If the name is empty, skip the search
        if not name:
            return None, 'name not found', None

        logging.debug(f"Searching for {row['Original Order']}: {name}")

        # Split the name into parts (genus, species, subspecies)
        genus, species, subspecies = split_species_name(name)
        logging.debug(f"Splitting  {name} into {genus}, {species}, {subspecies}")

        # Start by searching for the genus
        if genus:
            genus_match = taxon_df[taxon_df['dwc:genericName'].str.lower() == genus.lower()]
            if not genus_match.empty:
                logging.debug(f"Found match for {genus}")
                # If genus is found, search for the species
                if species:
                    species_match = taxon_df[(taxon_df['dwc:genericName'].str.lower() == genus.lower()) &
                                             (taxon_df['dwc:specificEpithet'].str.lower() == species.lower())]
                    if not species_match.empty:
                        logging.debug(f"Found match for {species} in {genus}")
                        # If species is found, search for the subspecies
                        if subspecies:
                            subspecies_match = taxon_df[(taxon_df['dwc:genericName'].str.lower() == genus.lower()) &
                                                        (taxon_df['dwc:specificEpithet'].str.lower() == species.lower()) &
                                                        (taxon_df['dwc:infraspecificEpithet'].str.lower() == subspecies.lower())]
                            if not subspecies_match.empty:
                                logging.debug(f"Found match for {subspecies} in {species}")
                                return subspecies_match.iloc[0]['dwc:taxonID'], 'subspecies', None
                            else:
                                return species_match.iloc[0]['dwc:taxonID'], 'species', None
                        else:
                            return species_match.iloc[0]['dwc:taxonID'], 'species', None
                    else:
                        return genus_match.iloc[0]['dwc:taxonID'], 'Genus Matched. Species not found', name
                else:
                    return genus_match.iloc[0]['dwc:taxonID'], 'genus', None
            else:
                return None, 'genus not found', name

        return None, 'genus not found', None
        
    except Exception as e:
        logging.error(f"Error while searching for {name}: {str(e)}")
        return None, 'error', str(e)




# Apply the search function to the DataFrame
df['Taxon ID'], df['Match Level'], df['Error'] = zip(*df.apply(search_species, axis=1))
df['Search Result'] = df.apply(search_species, axis=1)
df['dwc:taxonID'], df['Match Level'], df['Error'] = zip(*df['Search Result'].apply(lambda x: x if x else (None, 'unmatched', None)))
df = df.drop(columns=['Search Result'])

logging.info("Applied search function to the DataFrame.")

# Create 'Unmatched Names' column
#df['Unmatched Names'] = df['Bi or Trinomial Name'].where(df['Match Level'].isna(), '')
# create a new column to store unmatched names
df['Unmatched Names'] = df[df['Match Level'] == 'unmatched']['Bi or Trinomial Name']

# After applying the search function to 'df' and updating 'Unmatched Names', write 'df' to a CSV file
df.to_csv('matched_names_and_taxon_ids.csv', index=False)


# Merge the DataFrames to include distribution data
merged_df = df.merge(distribution_df, left_on='Taxon ID', right_on='dwc:taxonID', how='left')
merged_df = merged_df.drop_duplicates(subset=['Original Order'], keep='first', inplace=False,  ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_file = os.path.join(output_folder, "merged_output.csv")
merged_df.to_csv(output_file, index=False)
logging.info("Saved merged DataFrame to merged_output.csv.")

# create empty 'Countries' column
merged_df['Countries'] = ''


"""
def extract_countries(row):
    try:
        # Get the locality information for the taxon
        locality = row['dwc:locality'] if pd.notna(row['dwc:locality']) else ''

        # Replace common abbreviations
        locality = locality.replace('USA', 'United States').replace('UK', 'United Kingdom').replace('UAE', 'United Arab Emirates')

        # Remove data within parentheses
        locality = re.sub(r'\(.*?\)', '', locality)

        # Use GeoText to identify the countries in the locality string
        places = GeoText(locality)
        countries = list(places.countries)

        # If no valid country name is found, return a default value
        if not countries:
            return 'No Valid Country Found'

        # Join the country names with commas and return the result
        return ', '.join(countries)

    except Exception as e:
        logging.error(f"Error while extracting countries from {row['dwc:locality']}: {str(e)}")
        return 'Error'

# Add a new 'Countries' column to the DataFrame
merged_df['Countries'] = merged_df.apply(extract_countries, axis=1)
"""
"""
import re
from geotext import GeoText

def extract_countries(row):
    try:
        # Get the locality information for the taxon
        locality = row['dwc:locality'] if pd.notna(row['dwc:locality']) else ''

        # Replace common abbreviations
        locality = locality.replace('USA', 'United States').replace('UK', 'United Kingdom').replace('UAE', 'United Arab Emirates')

        # Remove data within parentheses
        locality = re.sub(r'\(.*?\)', '', locality)

        # Use GeoText to identify the countries in the locality string
        places = GeoText(locality)
        countries = list(places.countries)

        # If no valid country name is found, return a default value
        if not countries:
            return 'No Valid Country Found'

        # Join the country names with commas and return the result
        return ', '.join(countries)

    except Exception as e:
        logging.error(f"Error while extracting countries from {row['dwc:locality']}: {str(e)}")
        return 'Error'

# Add a new 'Countries' column to the DataFrame
merged_df['Countries'] = merged_df.apply(extract_countries, axis=1)
"""


# Load the geonamescache country data
gc = geonamescache.GeonamesCache()
countries = gc.get_countries_by_names()

# Extract country names from 'dwc:locality'
def extract_countries(row, countries):
    try:
        # Get the locality information for the taxon
        locality = row['dwc:locality'] if pd.notna(row['dwc:locality']) else ''

        # Use SpaCy to identify the geographic locations in the locality string
        doc = nlp(locality)
        locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']

        # For each location, check if it's a valid country name
        valid_country_names = [country['name'].lower() for country in countries.values()]
        valid_countries = set(valid_country_names)
        countries = [location for location in locations if location.lower() in valid_countries]

        # If no valid country name is found, return a default value
        if not countries:
            return 'No Valid Country Found'
        #log country names
        logging.debug(f"Country names found in {locality}: {countries}")

        # Join the country names with commas and return the result
        return ', '.join(countries) 
    except Exception as e:
        logging.error(f"Error while extracting countries from {row['dwc:locality']}: {str(e)}")
        return 'Error'

# Add a new 'Countries' column to the DataFrame
merged_df['Countries'] = merged_df.apply(lambda row: extract_countries(row, countries), axis=1)
# Extract only the first country name from the 'dwc:locality' column
merged_df['Country Name'] = merged_df['Countries'].str.split(r'[,(]').str[0]
#export database with country information to batch.csv CSV file
merged_df.to_csv('batch.csv', index=False)
logging.info("Saved merged DataFrame to batch.csv.")

# Now, you can proceed with the write back to Google Sheets
batch_update_values = []

#for index, row in merged_df.sort_values('Original Order').iterrows():
    ## Skip rows with order values that have been used
    #if row['Original Order'] in used_row_orders:
        #continue
    #used_row_orders.add(row['Original Order'])


for index, row in merged_df.iterrows():
    cleaned_name = '' if pd.isna(row['Cleaned Name']) else str(row['Cleaned Name'])
    original_order = '' if pd.isna(row['Original Order']) else str(row['Original Order'])
    taxon_id = '' if pd.isna(row['Taxon ID']) else str(row['Taxon ID'])
    match_level = '' if pd.isna(row['Match Level']) else str(row['Match Level'])
    unmatched_names = '' if pd.isna(row['Unmatched Names']) else str(row['Unmatched Names'])
    locality = '' if pd.isna(row['dwc:locality']) else str(row['dwc:locality'])
    country_name = '' if pd.isna(row['Country Name']) else str(row['Country Name'])
    countries = '' if pd.isna(row['Countries']) else str(row['Countries'])
    error = '' if pd.isna(row['Error']) else str(row['Error'])

    # Append the row values to the batch update
    batch_update_values.append([cleaned_name, original_order, taxon_id, match_level, unmatched_names, locality, country_name, countries, error])

# Get the total number of columns in the sheet
num_cols = len(sheet.get_all_values()[0])


# Start from 41st column if there are more than 40 columns
start_col = 41 if num_cols > 40 else num_cols + 1

# convert column numbers to their corresponding letters in A1 notation (for example, 1 -> 'A', 2 -> 'B', 26 -> 'Z', 27 -> 'AA', etc.)
def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string
# Convert the start and end column numbers to A1 notation
start_col_letter = colnum_string(start_col)
new_cols = ['Cleaned Name', 'Original Order', 'Taxon ID', 'Match Level', 'Unmatched Names', 'Locality', 'Country Name', 'Countries', 'Error']
end_col_letter = colnum_string(start_col + len(new_cols) - 1)

# Prepare the range for batch update
start_row = 5  # starting from the 5th row
end_row = start_row + len(batch_update_values) - 1  # the last row to be updated

# Define the range in A1 notation
range_ = f'{start_col_letter}{start_row}:{end_col_letter}{end_row}'
logging.info(f'Range for batch update: {range_}')

# Add new columns to Google Sheets if not already present
headers = sheet.row_values(3)  # Assuming headers are in 3rd row
for i, col in enumerate(new_cols, start=start_col):
    col_letter = colnum_string(i)
    if col not in headers:
        sheet.update_cell(3, i, col)

               
import csv
import time
import os

logging.info('Starting to update the values in Google Sheet')

# Prepare the 'requests' list for the batch update
range_name = f'Names_Src_Details!{start_col_letter}{start_row}:{end_col_letter}{end_row}'

# Get the spreadsheet instead of the worksheet
spreadsheet = client.open_by_key(sheet_id)
worksheet = spreadsheet.worksheet("Names_Src_Details")  # Assuming the worksheet name is "Names_Src_Details"

# Save batch update values to CSV file
csv_file_path = os.path.join(os.getcwd(), "batch_update_values.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(batch_update_values)

logging.info(f"Batch update values saved to {csv_file_path}.")

# Execute the batch update with retries
max_retries = 5
for attempt in range(max_retries):
    try:
        sheet.update(range_, batch_update_values)
        logging.info("Successfully updated the Google Sheet.")
        break
    except Exception as e:
        logging.error(f"Error while updating the Google Sheet on attempt {attempt + 1}: {str(e)}")
        if attempt < max_retries - 1:  # i.e. If this is not the last attempt
            time.sleep(5)  # Wait for 5 seconds before next attempt
        else:
            logging.error(f"Failed to update the Google Sheet after {max_retries} attempts.")

# Log the script execution end
logging.info('Primary execution finished! Analysing Results!')


# Define a function to analyze the results
def analyze_results(df):
    total_rows = len(df)
    matched_rows = len(df[df['Match Level'] != 'not found'])
    error_rows = len(df[df['Match Level'] == 'error'])
    
    logging.info(f"Total rows: {total_rows}")
    logging.info(f"Matched rows: {matched_rows}")
    logging.info(f"Error rows: {error_rows}")
    logging.info(f"Match rate: {matched_rows / total_rows * 100:.2f}%")
    logging.info(f"Error rate: {error_rows / total_rows * 100:.2f}%")

# Call the function at the end of the script
analyze_results(merged_df)

logging.info("Analyzed the results.")
# If there are errors, save them to a separate CSV file
error_df = merged_df[merged_df['Match Level'] == 'error']
if not error_df.empty:
    error_file = os.path.join(output_folder, "errors.csv")
    error_df.to_csv(error_file, index=False)
    logging.info("Saved errors to errors.csv.")

# Log the script execution end
logging.info('Script execution finished!')
