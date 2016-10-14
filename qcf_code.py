from pandas import DataFrame, read_sas
import pandas as pd

# Read in SAS data set - takes a while...
data = read_sas("crsp_comp.sas7bdat")

# List column names
# list(data.columns.values)

# Sort the set by cusip, permno, cik, and then year (descending)
sorted = data.sort_values(['CUSIP', 'PERMNO', 'cik', 'year'], ascending=[True, True, True, False])

# Pick out the most recent unique cik values

# Iterate over cik values and download the 10K for each over 1995 - 2008

# Parse each 10K and do sentiment analysis

# Develop and run model