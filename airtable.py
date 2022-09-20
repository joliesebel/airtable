"""airtable.py: Updates information from dash files/dash applications/rank calculations/photos folder to required airtable databases"""
__author__ = "Jolie Sebel"

#Import necessary packages
import os
import calendar
import pandas as pd
import numpy as np
from pyairtable import Table
from datetime import date, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

"""Gets the airtable base given an abbreviation"""
def get_base(abbr):
    #If the abbreviation is AM then the airtable base is Amazon
    if abbr == 'CA' or abbr == 'MX' or abbr == 'US' or abbr == 'NOAMT':
        base = 'NOAM'

    #If the abbreviation is AMSP then the airtable base is Amazon - SP
    elif abbr == 'SPK':
        base = 'Saturday Park'

    #Otherwise the base is EMEA
    else:
        base = 'EMEA'

    #Return the base value
    return base

"""Gets the table_name given an abbreviation and a date"""
def get_table_name(abbr, dt):
    #Extract the year and store the stringed value of the year
    year = dt.year
    year_str = str(year)

    #If the abbreviation is Amazon or Amazon Saturday Park the table name is just the year
    if abbr == 'SPK':
        table_name = year_str

    #If the abbreviation stands for a country then the table name is the abbreviation plus the last two characters of the year
    else:
        table_name = abbr + year_str[len(year_str) - 2:]

    #Return the table name
    return table_name

"""Extracts airtable values and returns the table object with table_name from the airtable base"""
def airtable_get_table(base, table_name):
    #Extract the enviroment variable storing the base id given the base
    if base == 'NOAM':
        base_id = os.environ.get('AIRTABLE_BASE_ID_NOAM')
    elif base == 'Saturday Park':
        base_id = os.environ.get('AIRTABLE_BASE_ID_SATURDAY_PARK')
    elif base == 'Ecomm':
        base_id = os.environ.get('AIRTABLE_BASE_ID_ECOMM')
    elif base == 'EMEA':
        base_id = os.environ.get('AIRTABLE_BASE_ID_EMEA')
    elif base == 'Photo Queue':
        base_id = os.environ.get('AIRTABLE_BASE_ID_PHOTO_QUEUE')
    elif base == 'Image Title':
        base_id = os.environ.get('AIRABTABLE_BASE_ID_IMAGE_TITLE')

    #Extract the enviroment variable storing the API Key
    api_key = os.environ.get('AIRTABLE_API_KEY')

    #Using this information get and return the table object
    table = Table(api_key, base_id, table_name)
    return table

"""Makes a request to Airtable for all records from a single table and converts it to a Pandas dataframe"""
def airtable_download(table):
    #Get all the records from the table object in the form of a dictonary
    airtable_records = table.all()

    #Convert the dictonary to a Pandas dataframe with columns id and all other fields
    airtable_rows = []
    for record in airtable_records:
        row = {'id': record['id']} | record['fields']
        airtable_rows.append(row)
    airtable_dataframe = pd.DataFrame(airtable_rows)

    #Return the airtable dataframe
    return airtable_dataframe

"""Get the corresponding query SellerID notation given the abbreviation"""
def get_seller_id(abbr):
    #Return the correct SellerID notation based on abbr
    match abbr:
        #Canada
        case 'CA':
            return 'SellerID = 1'
        #Mexico
        case 'MX':
            return 'SellerID = 2'
        #United States
        case 'US':
            return 'SellerID = 3'
        #North America Total
        case 'NOAMT':
            return '(SellerID = 1 OR SellerID = 2 OR SellerID = 3)'
        #France
        case 'FR':
            return 'SellerID = 5'
        #Netherlands
        case 'NL':
            return 'SellerID = 6'
        #Poland
        case 'PL':
            return 'SellerID = 7'
        #United Kingdom
        case 'UK':
            return 'SellerID = 8'
        #Germany
        case 'GR':
            return 'SellerID = 9'
        #Spain
        case 'SP':
            return 'SellerID = 10'
        #Sweeden
        case 'SW':
            return 'SellerID = 11'
        #Italy
        case 'IT':
            return 'SellerID = 13'
        #EU Total (Every Country But the United Kingdom)  
        case 'EUT':
            return '(SellerID = 5 OR SellerID = 6 OR SellerID = 7 OR SellerID = 9 OR SellerID = 10 OR SellerID = 11 OR SellerID = 13)'
        #EMEA Total (Every Country)
        case 'EMEAT':
            return '(SellerID = 5 OR SellerID = 6 OR SellerID = 7 OR SellerID = 8 OR SellerID = 9 OR SellerID = 10 OR SellerID = 11 OR SellerID = 13)'
        #Saturday Park
        case 'SPK':
            return 'SellerID = 14'

"""Downloads the weekly data from SQL query into Pandas dataframe"""
def week_download(abbr, dt):
    #Calculate the start and end of week
    start = dt - timedelta(days = dt.isoweekday() % 7)
    end = start + timedelta(days = 6)

    #Make sure start date is not last year
    if(start.year != dt.year):
        start = date(dt.year, 1, 1)

    #Make sure end date is not next year
    if(end.year != dt.year):
        end = date(dt.year, 12, 31)

    #Create the string representing the column name in airtable
    week_name = str(start.month) + '/' + str(start.day) + ' - ' + str(end.month) + '/' + str(end.day)

    #Get SellerID field
    sellerId = get_seller_id(abbr)

    #Create the string representing the query with given week_name and start and end dates
    qry = f'''
    SELECT
        ASIN,
        SUM(Amount) AS '{week_name} Sales',
        SUM(Quantity) AS '{week_name} Units'
    FROM
        jayfranco.mws_orders_metric
    WHERE
        {sellerId}
        AND dtPurchasedOn BETWEEN '{start}' AND '{end}'
    GROUP BY
        ASIN
    '''
    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_APP_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    week_dataframe = pd.read_sql_query(qry,dash_engine)
    return week_dataframe

"""Downloads the monthly data from SQL query into Pandas dataframe"""
def month_download(abbr, dt):
    #Get the month and the year from today's date
    month = dt.month
    year = dt.year

    #Get the string representing the month name to represent the data in airtable
    month_name = calendar.month_name[month]

    #Get SellerID field
    sellerId = get_seller_id(abbr)

    #Create the string representing the query with month_name and month and year
    qry = f'''
    SELECT
        ASIN,
        SUM(Amount) AS '{month_name} Sales',
        SUM(Quantity) AS '{month_name} Units'
    FROM
        jayfranco.mws_orders_metric
    WHERE
        {sellerId}
        AND MONTH(dtPurchasedOn) = {month}
        AND YEAR(dtPurchasedOn) = {year}
    GROUP BY
        ASIN
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_APP_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    month_dataframe = pd.read_sql_query(qry,dash_engine)
    return month_dataframe

"""Downloads the quarterly data from SQL query into Pandas dataframe"""
def quarter_download(abbr, dt):
    #Get the month and the year from today's date
    month = dt.month
    year = dt.year

    #Calculate the quauter this month is in
    quarter = (month-1)//3 + 1

    #Get SellerID field
    sellerId = get_seller_id(abbr)

    #Create the string representing the query with quater and year
    qry = f'''
    SELECT
        ASIN,
        SUM(Amount) AS 'Quarter {quarter} Sales',
        SUM(Quantity) AS 'Quarter {quarter} Units'
    FROM
        jayfranco.mws_orders_metric
    WHERE
        {sellerId}
        AND QUARTER(dtPurchasedOn) = {quarter}
        AND YEAR(dtPurchasedOn) = {year}
    GROUP BY
        ASIN
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_APP_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    month_dataframe = pd.read_sql_query(qry,dash_engine)
    return month_dataframe

"""Downloads the year data from SQL query into Pandas dataframe"""
def year_download(abbr, dt):
    #Get the year from today's date
    year = dt.year

    #Get SellerID field
    sellerId = get_seller_id(abbr)

    #Create the string representing the query with year
    qry = f'''
    SELECT
        ASIN,
        SUM(Amount) AS 'YTD Sales',
        SUM(Quantity) AS 'YTD Units'
    FROM
        jayfranco.mws_orders_metric
    WHERE
        {sellerId}
        AND YEAR(dtPurchasedOn) = {year}
    GROUP BY
        ASIN
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_APP_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    year_dataframe = pd.read_sql_query(qry,dash_engine)
    return year_dataframe

"""Downloads the inventory health data from SQL query into Pandas dataframe"""
def inventory_health_download():
    #Create the string representing the query with year
    qry = '''
    SELECT
	    ASIN,
	    InvAge0to90Days as 'Inventory Age 0-90',
	    InvAge91to180Days as 'Inventory Age 91-180',
	    InvAge181to270Days as 'Inventory Age 181-270',
	    InvAge271to365Days as 'Inventory Age 271-365',
	    InvAge365PlusDays as 'Inventory Age 365+'
    FROM
	    jayfranco.mws_inventory_health
    GROUP BY 
	    ASIN
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_APP_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    inventory_health_dataframe = pd.read_sql_query(qry,dash_engine)
    return inventory_health_dataframe

"""Calculates the inter rankings for each SKU and returns the values as a Pandas Dataframe"""
def inter_rank_calculate(airtable_yearly_dataframe):
    rank_dataframe = pd.DataFrame()

    if ('YTD Units' in airtable_yearly_dataframe.columns):
        #Extract relevant airtable information from airtable dataframe
        rank_dataframe = airtable_yearly_dataframe[['ASIN', 'YTD Units', 'Property Lookup', 'Product Type Lookup', 'Item Type Lookup']]

        #Convert list types to string types
        rank_dataframe = rank_dataframe.astype({'Property Lookup': 'string', 'Product Type Lookup': 'string', 'Item Type Lookup': 'string'})

        #Get rid of any null values
        rank_dataframe = rank_dataframe[rank_dataframe['YTD Units'].notnull() & rank_dataframe['Property Lookup'].notnull() & rank_dataframe['Product Type Lookup'].notnull() & rank_dataframe['Item Type Lookup'].notnull()]

        #Calculate rankings within property
        rank_dataframe['Inter Property Rank'] = rank_dataframe.groupby('Property Lookup')['YTD Units'].rank(ascending = False)
        
        #Calculate rankings within product type 
        rank_dataframe['Inter Product Rank'] = rank_dataframe.groupby('Product Type Lookup')['YTD Units'].rank(ascending = False)

        #Calculate rankings within item type
        rank_dataframe['Inter Item Rank'] = rank_dataframe.groupby('Item Type Lookup')['YTD Units'].rank(ascending = False)

        #Calculate rankings within property x product type
        rank_dataframe['Inter Property x Product Rank'] = rank_dataframe.groupby(['Property Lookup', 'Product Type Lookup'])['YTD Units'].rank(ascending = False)

        #Calculate rankings within property x item type
        rank_dataframe['Inter Property x Item Rank'] = rank_dataframe.groupby(['Property Lookup', 'Item Type Lookup'])['YTD Units'].rank(ascending = False)

        #Drop unessasary columns
        rank_dataframe = rank_dataframe.drop(columns = ['YTD Units','Property Lookup', 'Product Type Lookup', 'Item Type Lookup'])

    #Return the dataframe representing the values with each of their rankings
    return rank_dataframe

"""Calculates the rankings for each property and returns the values as a Pandas Dataframe"""
def property_rank_calculate(airtable_yearly_dataframe, table_name):
    #Determine which months unit columns are in airtable_yearly_dataframe
    relevant_columns = ['Property Lookup']
    for month in range(1, 12):
        month_name = calendar.month_name[month]
        units_name = month_name + ' Units'
        if units_name in airtable_yearly_dataframe:
            relevant_columns.append(units_name)

    #Extract relevant airtable dataframe columns and edit property lookup field for comparison
    compare_dataframe = airtable_yearly_dataframe[relevant_columns]
    compare_dataframe = compare_dataframe.astype({'Property Lookup': 'string'})
    compare_dataframe['Property Lookup'] = compare_dataframe['Property Lookup'].str.slice(2, -2)

    #Calculate the totals for each month
    total_dataframe = compare_dataframe.groupby('Property Lookup').sum()
    total_dataframe = total_dataframe.loc[:, (total_dataframe != 0).any(axis = 0)]

    #Calculate the rankings based on the total monthly units
    rank_dataframe = total_dataframe.rank(ascending = False)

    #Rename the ranking columns to correct names
    for column in rank_dataframe.columns:
        rename_str = column.replace('Units', 'Rank ')
        rename_str = rename_str + table_name
        rank_dataframe = rank_dataframe.rename(columns = {column: rename_str})

    #Make property lookup index a column and rename it to property
    rank_dataframe.reset_index(inplace=True)
    rank_dataframe = rank_dataframe.rename(columns = {'Property Lookup':'Property'})

    #Return rank dataframe
    return rank_dataframe

"""Extracts the ON_HAND Saturday Park inventory from dash file database and returns it as a Pandas dataframe"""
def ecom_sp_avaliable_inventory_download():
    #Create the string representing the query for all inventory greater than zerp
    qry = f'''
    SELECT
	    `STYLE` as 'JF Style Number',
	    ON_HAND as 'Ecom Available Inventory'
    FROM
	    inventory.dash03sp
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_FILE_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    avaliable_inventory_dataframe = pd.read_sql_query(qry,dash_engine)

    #Return Pandas dataframe
    return avaliable_inventory_dataframe

"""Reads in fba inventory query"""
def fba_inventory_download():
    #Create the string representing the query for all inventory greater than zerp
    qry = f'''
    SELECT
	    ASIN,
	    SUM(totalQuantity) as 'FBA Inventory'
    FROM
	    inventory.fba_inventory
    GROUP BY
	    ASIN
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_FILE_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    fba_inventory_dataframe = pd.read_sql_query(qry,dash_engine)

    #Return fba inventory dataframe
    return fba_inventory_dataframe

"""Reads in fba inventory query for all inventory greater than zero"""
def fba_timestamp_download():
    #Create the string representing the query for all inventory greater than zerp
    qry = f'''
    SELECT
	    ASIN
    FROM
	    inventory.fba_inventory
    WHERE
	    totalQuantity > 0
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_FILE_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    fba_timestamp_dataframe = pd.read_sql_query(qry,dash_engine)

    #Create a json date value with today
    dt = date.today()
    json_dt = dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    #Set the FBA Inventory Timestamp column to that date
    fba_timestamp_dataframe['FBA Inventory Timestamp'] = json_dt

    #Return Pandas dataframe
    return fba_timestamp_dataframe

"""Reads in last recieved date and last shipped date"""
def last_date_download():
    #Create the string representing the query for all inventory greater than zerp
    qry = f'''
    SELECT
	    `STYLE` as 'JF Style Number',
	    LASTRCVDT as 'Last Received Date',
	    LASTSHPDT as 'Last Ship Date'
    FROM
	    inventory.dash03
    '''

    #Run the query and return it as a Pandas dataframe
    conn_mysql = os.environ.get('DASH_FILE_QUERY_STR')
    dash_engine = create_engine(conn_mysql, echo=False)
    last_date_dataframe = pd.read_sql_query(qry,dash_engine)

    #Return Pandas dataframe
    return last_date_dataframe

"""Reads in last dates as potential first dates and returns them as a Pandas dataframe"""
def first_date_download(last_date_dataframe):
    #Set first date dataframe equal to last date dataframe
    first_date_dataframe = last_date_dataframe

    #Rename columns first date names
    first_date_dataframe = first_date_dataframe.rename(columns = {'Last Received Date':'First Received Date', 'Last Ship Date':'First Ship Date'})

    #Return first date dataframe
    return first_date_dataframe

"""Compares airtable_df and new_df and returns a Pandas Dataframe of differences for updating"""
def df_find_diff(airtable_df, new_df, time = False, timestamp = False):
    #Initalize returned dataframe to an empty Pandas Dataframe
    df = pd.DataFrame()

    #If new dataframe is not empty compare dataframes
    if not new_df.empty:
        #Extract columns of new_df
        cols = new_df.columns

        #Extract list of columns not including the key
        keyless_cols = cols[1:]

        #Create a variable that store the key or the first column on new_df
        key = cols[0]

        #Create a list containing the id plus the cols of new_df
        id_cols = cols.insert(0, 'id')

        #For each column in new_df if it does not exist in airtable_df add it with null values
        for col in cols:
            if col not in airtable_df.columns:
                airtable_df[col] = np.nan

        #If not comparing times then check for incorrect null values (the reason I have this is because Im having the numbers set to zero and I'm not sure what the defualt date would be)
        if time == False:
            #Extract keys not in new_df that are in airtable_df
            nulls_df = airtable_df[id_cols]
            nulls_df = nulls_df[-airtable_df[key].isin(new_df[key])]

            #If checking just for saturday park inventory only make sure to account for that
            if key == 'JF Style Number':
                nulls_df = nulls_df.astype({key: 'string'})
                airtable_df = airtable_df.astype({key: 'string'})
                nulls_df = nulls_df[nulls_df[key].str.contains('SP')]

            #From these values delete any rows that contain only null or zero vaues
            nulls_df = nulls_df[-(nulls_df[keyless_cols].isin([np.nan, 0]).all(axis = 1))]

            #Create a list of zeros for the number of keyless_cols
            zeros = [0 for _ in range(len(keyless_cols))]
            
            #Add zero rows to new_df for all values that are suppose to be null
            for index, row in nulls_df.iterrows():
                    key_zeros = [row[key]] + zeros
                    new_df.loc[len(new_df.index)] = key_zeros

        #Extract id and new_df columns from airtable_df for comparison
        airtable_compare_df = airtable_df[id_cols]

        #If comparing for timestamp purposes compare only with the null values of airtable_compare_df
        if timestamp == True:
            #Get all the values of airtable_df where keyless_cols is null (there is no date there)
            airtable_compare_df = airtable_compare_df[airtable_compare_df[keyless_cols].isin([np.nan]).any(axis = 1)]

        #Merge ids with new_df on the first column for comparison, CANNOT HAVE DUPLICATES MAKE SURE QUERY IS CORRECT
        airtable_id_key_df = airtable_compare_df[['id', key]]
        new_compare_df = new_df.merge(airtable_id_key_df, how = 'inner', on = key)

        #Reorder new_compare_df
        new_compare_df = new_compare_df[id_cols]

        #remove keys from airtable_compare_df that are not in new_compare_df
        airtable_compare_df = airtable_compare_df[airtable_compare_df[key].isin(new_compare_df[key])]

        #Make the ids of both dataframes the key
        airtable_compare_df = airtable_compare_df.set_index('id')
        new_compare_df = new_compare_df.set_index('id')

        #Sort both dataframes by their index
        airtable_compare_df = airtable_compare_df.sort_index()
        new_compare_df = new_compare_df.sort_index()

        #Compare the two dataframes
        df = airtable_compare_df.compare(new_compare_df)

        #Drop self columns from dataframe
        df = df[df.columns.drop(list(df.filter(regex='self')))]

        #Drop remaining header below column name headers
        df.columns = df.columns.droplevel(-1)

    #Return compare_df
    return df

"""Compares upc values between airtable dataframes and returns the dataframes to be uploaded to airtable"""
def upc_found_dfs(image_title_airtable, photo_queue_airtable):
    #Extract values where upc1 is not null
    upc1_no_nulls = photo_queue_airtable[photo_queue_airtable['UPC 1'].notna()]

    #Extract id into list                           
    id_list = [id for id in upc1_no_nulls['id']]

    #Extract upc dictonary pairs into a list
    upc_cols = ['UPC 1', 'UPC 2', 'UPC 3', 'UPC 4', 'UPC 5', 'UPC 6', 'UPC 7', 'UPC 8', 'UPC 9', 'UPC 10', 'UPC 11', 'UPC 12']
    upc_no_nulls = upc1_no_nulls[upc_cols]
    upc_dict_list = upc_no_nulls.values.tolist()

    #Extract just the upcs from the upc list
    upc_no_nan_pairs = [[d for d in pairs if str(d) != 'nan'] for pairs in upc_dict_list] 
    upc_pairs = [[d.get('text') for d in pairs] for pairs in upc_no_nan_pairs]

    #Create a dictonary of record ids and their corresponding upc pairs
    upc_id_pairs = {id_list[i]: upc_pairs[i] for i in range(len(id_list))}

    #Create a flat list of all the UPCS
    upc_flat = [item for sublist in upc_pairs for item in sublist]

    #Create a flat list of all the ids
    id_flat = [id for id in upc_id_pairs for i in range(len(upc_id_pairs.get(id)))]

    #Convert lists into a Pandas dataframe
    photo_queue_upload = pd.DataFrame(upc_flat, index = id_flat, columns = ['UPC'])

    #Extract image title values where Pair Found is null
    image_title_pairless = image_title_airtable[image_title_airtable['Pair Found'].isna()]

    #Add Photo Completed Column based on if the upcs are in image_title_airtable
    photo_queue_upload['Photo Completed'] = photo_queue_upload['UPC'].isin(image_title_pairless['UPC'])

    #Drop not true values
    photo_queue_upload = photo_queue_upload[photo_queue_upload['Photo Completed'] == True]

    #Remove duplicate index but keep the first
    photo_queue_upload = photo_queue_upload.groupby(photo_queue_upload.index).first()

    #Drop UPC column
    photo_queue_upload = photo_queue_upload.drop(columns = 'UPC')

    #Create list of UPCS that are found
    upcs_found_ids = photo_queue_upload.index.values.tolist()
    upcs_found = []
    for id in upcs_found_ids:
        upcs_found.extend(upc_id_pairs.get(id))

    #Create dataframe to be used for comparison for the upcs
    upcs_found_df = pd.DataFrame(upcs_found, columns = ['UPC'])
    upcs_found_df['Pair Found'] = True

    #Extract values from image title to be used for comparison
    image_title_compare = image_title_airtable[['id', 'UPC']]

    #Remove all values where pairs found is true
    image_title_compare = image_title_compare[image_title_airtable['Pair Found'].isna()]

    #Merge image title compare and upcs found
    image_title_upload = image_title_compare.merge(upcs_found_df, how = 'inner', on = 'UPC')

    #Set index equal to id
    image_title_upload = image_title_upload.set_index('id')

    #Remove duplicate index but keep the first
    image_title_upload = image_title_upload.groupby(image_title_upload.index).first()

    #Drop upc column
    image_title_upload = image_title_upload.drop(columns = 'UPC')

    #Return dataframes
    return photo_queue_upload, image_title_upload

"""Updates changes from diff_dataframe to airtable"""
def airtable_update_columns(diff_dataframe, table):
    #If the there are values in diff_dataframe update the current airtable values
    if not diff_dataframe.empty:
        #Print updating message
        print('Updating: ' + (', ').join(diff_dataframe.columns))

        #Convert diff_dataframe to a dictonary in index for where the key is the id and the values are the fields
        diff_dict = diff_dataframe.to_dict('index')

        #Create a list that will store values in the format needed for updating
        update_list = []

        #For each key, value in diff_dict add the corresponding dictonary to update_list
        for id, fields in diff_dict.items():
            clean_fields = {k: fields[k] for k in fields if pd.notna(fields[k])}
            update_list.append({'id': id, 'fields': clean_fields})
        
        #Update airtable using update_list
        table.batch_update(update_list)

"""Updates airtable with today's and past dash information based on the given abbreviation"""
def update_dash(abbr):
    #Get today's date
    today = date.today()

    #Get the base and table name given the abbreviation and today's year
    base = get_base(abbr)
    table_name = get_table_name(abbr, today)

    #Print message specifing what is being updated
    print(base + ': ' + table_name + ' Updates')

    #Get the relevant airtable table and its values as a Pandas dataframe
    table = airtable_get_table(base, table_name)
    airtable_dataframe = airtable_download(table)
    
    #Get today's weekly information from dash applications, find the differences between this information and airtable, and upload them
    week_dataframe = week_download(abbr, today)
    week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
    airtable_update_columns(week_diff_dataframe, table)

    #Get today's monthly information from dash applications, find the differences between this information and airtable, and upload them
    month_dataframe = month_download(abbr, today)
    month_diff_dataframe = df_find_diff(airtable_dataframe, month_dataframe)
    airtable_update_columns(month_diff_dataframe, table)

    #Get today's quarterly information from dash applications, find the differences between this information and airtable, and upload them
    quarter_dataframe = quarter_download(abbr, today)
    quarter_diff_dataframe = df_find_diff(airtable_dataframe, quarter_dataframe)
    airtable_update_columns(quarter_diff_dataframe, table)

    #Get today's yearly information from dash applications, find the differences between this information and airtable, and upload them
    year_dataframe = year_download(abbr, today)
    year_diff_dataframe = df_find_diff(airtable_dataframe, year_dataframe)
    airtable_update_columns(year_diff_dataframe, table)

    #Update past information from dash applications to airtable
    update_dash_past(abbr, today, table, airtable_dataframe)

"""Runs a full update from dash applications on all columns of year table of airtable"""
def update_dash_full(abbr, year):
    #Create a date variable the stores the first of year
    dt = date(year, 1, 1)

    #Get the base and table name given the abbreviation and today's year
    base = get_base(abbr)
    table_name = get_table_name(abbr, dt)

    #Get the relevant airtable table and its values as a Pandas dataframe
    table = airtable_get_table(base, table_name)
    airtable_dataframe = airtable_download(table)

    #Print message specifing what is being updated
    print(base + ': ' + table_name + ' Updates')

    #Get date's yearly information from dash applications, find the differences between this information and airtable, and upload them
    year_dataframe = year_download(abbr, dt)
    year_diff_dataframe = df_find_diff(airtable_dataframe, year_dataframe)
    airtable_update_columns(year_diff_dataframe, table)

    #Iterate through quaters of the year and update monthly information
    for month in range(1, 12, 3):
        #Create variable for the first day of the month
        dt = date(year, month, 1)

        #Get today's quarterly information from dash applications, find the differences between this information and airtable, and upload them
        quarter_dataframe = quarter_download(abbr, dt)
        quarter_diff_dataframe = df_find_diff(airtable_dataframe, quarter_dataframe)
        airtable_update_columns(quarter_diff_dataframe, table)

    #Iterate through months of the year and update monthly information
    for month in range(1, 12):
        #Create variable for the first day of the month
        dt = date(year, month, 1)

        #Get date's monthly information from dash applications, find the differences between this information and airtable, and upload them
        month_dataframe = month_download(abbr, dt)
        month_diff_dataframe = df_find_diff(airtable_dataframe, month_dataframe)
        airtable_update_columns(month_diff_dataframe, table)

    #Iterate through weeks of the of the year and update weekly information
    dt = date(year, 1, 1)
    while(dt.year != year + 1):
        #Get date's weekly information from dash applications, find the differences between this information and airtable, and upload them
        week_dataframe = week_download(abbr, dt)
        week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
        airtable_update_columns(week_diff_dataframe, table)

        #Get next sunday's date
        dt += timedelta(days = 7)

"""Updates any past changes from dash applications to airtable"""
def update_dash_past(abbr, today, table, airtable_dataframe):
    #Get data from two weeks ago from today
    two_weeks_ago = today - timedelta(days = 14)

    #Update two weeks ago if in the same table
    if(two_weeks_ago.year == today.year):
        week_dataframe = week_download(abbr, two_weeks_ago)
        week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
        airtable_update_columns(week_diff_dataframe, table)
    
    #Get the date a week ago from today
    week_ago = today - timedelta(days = 7)

    #Calculate quarter of today and week_ago
    quarter_today = (today.month-1)//3 + 1
    quarter_week_ago = (week_ago.month-1)//3 + 1

    #If it is the end of the year update all of last years information
    if(week_ago.year != today.year):
        update_dash_full(abbr, week_ago.year)

    #Otherwise if it is the end of the quarter update last quarter, last month, and last week's information
    elif(quarter_today != quarter_week_ago):
        #Get week ago's quarterly information from dash applications, find the differences between this information and airtable, and upload them
        quarter_dataframe = quarter_download(abbr, week_ago)
        quarter_diff_dataframe = df_find_diff(airtable_dataframe, quarter_dataframe)
        airtable_update_columns(quarter_diff_dataframe, table)

        #Get week ago's monthly information from dash applications, find the differences between this information and airtable, and upload them
        month_dataframe = month_download(abbr, week_ago)
        month_diff_dataframe = df_find_diff(airtable_dataframe, month_dataframe)
        airtable_update_columns(month_diff_dataframe, table)

        #Get week ago's weekly information from dash applications, find the differences between this information and airtable, and upload them
        week_dataframe = week_download(abbr, week_ago)
        week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
        airtable_update_columns(week_diff_dataframe, table)

    #Otherwise if it is the end of the month update last month and last week's information
    elif(week_ago.month != today.month):
        #Get week ago's monthly information from dash applications, find the differences between this information and airtable, and upload them
        month_dataframe = month_download(abbr, week_ago)
        month_diff_dataframe = df_find_diff(airtable_dataframe, month_dataframe)
        airtable_update_columns(month_diff_dataframe, table)

        #Get week ago's weekly information from dash applications, find the differences between this information and airtable, and upload them
        week_dataframe = week_download(abbr, week_ago)
        week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
        airtable_update_columns(week_diff_dataframe, table)

    #Otherwise just update weekly information update weekly 
    else:
        #Get week ago's weekly information from dash applications, find the differences between this information and airtable, and upload them
        week_dataframe = week_download(abbr, week_ago)
        week_diff_dataframe = df_find_diff(airtable_dataframe, week_dataframe)
        airtable_update_columns(week_diff_dataframe, table)

"""Updates ranks based on the information stored in airtable"""
def update_rank(abbr, year = date.today().year):
    #Create a date variable the stores the first of year
    dt = date(year, 1, 1)

    #Get the base and table name given the abbreviation and today's year
    base = get_base(abbr)
    table_name = get_table_name(abbr, dt)

    #Get the yearly airtable table and its values as a Pandas dataframe
    yearly_table = airtable_get_table(base, table_name)
    airtable_yearly_dataframe = airtable_download(yearly_table)

    #Calculate the rankings of each ASINs, find the differences between this information and airtable, and upload them
    inter_rank_dataframe = inter_rank_calculate(airtable_yearly_dataframe)
    inter_rank_diff_dataframe = df_find_diff(airtable_yearly_dataframe, inter_rank_dataframe)
    airtable_update_columns(inter_rank_diff_dataframe, yearly_table)

    #Extract the property table and convert it to a Pandas Dataframe
    property_table = airtable_get_table(base, 'Properties')
    airtable_property_dataframe = airtable_download(property_table)

    #Calculate the property ranks from the airtable_yearly_dataframe
    property_rank_dataframe = property_rank_calculate(airtable_yearly_dataframe, table_name)

    #Find the differences between airtable_property_dataframe and property_dataframe
    property_rank_diff_dataframe = df_find_diff(airtable_property_dataframe, property_rank_dataframe)
    
    #Upload differences to property_table
    airtable_update_columns(property_rank_diff_dataframe, property_table)

"""Uploads ecom avaliable inventory changes to airtable"""
def update_ecomm_ecom_styles():
    #Print message specifing what is being updated
    print('Ecomm: Ecom Style Updates')

    #Extract Ecom Style Table and convert data to Pandas dataframe
    table = airtable_get_table('Ecomm', 'Ecom Styles')
    airtable_dataframe = airtable_download(table)

    #Extract Saturday Park ecom avaliable inventory dataframe, find differences, and upload them to airtable
    ecom_sp_inventory_dataframe = ecom_sp_avaliable_inventory_download()
    ecom_sp_inventory_diff_dataframe = df_find_diff(airtable_dataframe, ecom_sp_inventory_dataframe)
    airtable_update_columns(ecom_sp_inventory_diff_dataframe, table)

    #Extract last ship and recieved dates from dash files, find differences, and upload them to airtable
    last_date_dataframe = last_date_download()
    last_diff_dataframe = df_find_diff(airtable_dataframe, last_date_dataframe, True)
    airtable_update_columns(last_diff_dataframe, table)

    #Extract first shipped and recieved dates from last date dataframe, find differences, and upload them to airtable
    first_date_dataframe = first_date_download(last_date_dataframe)
    first_diff_dataframe = df_find_diff(airtable_dataframe, first_date_dataframe, True, True)
    airtable_update_columns(first_diff_dataframe, table)
    
"""Uploads fba inventory timestamp changes to airtable"""
def update_ecomm_amazon():
    #Print message specifing what is being updated
    print('Ecomm: Amazon Updates')

    #Extract Amazon Table and convert data to Pandas dataframe
    table = airtable_get_table('Ecomm', 'Amazon')
    airtable_dataframe = airtable_download(table)

    #Extract FBA inventory dataframe, find differences, and upload them to airtable
    fba_inventory_dataframe = fba_inventory_download()
    fba_inventory_diff_dataframe = df_find_diff(airtable_dataframe, fba_inventory_dataframe)
    airtable_update_columns(fba_inventory_diff_dataframe, table)
   
    #Extract FBA inventory timestamp dataframe, find differences, and upload them to airtable
    fba_timestamp_dataframe = fba_timestamp_download()
    fba_timestamp_diff_dataframe =  df_find_diff(airtable_dataframe, fba_timestamp_dataframe, True, True)
    airtable_update_columns(fba_timestamp_diff_dataframe, table)

    #Extract invenotry health dataframe, find differences, and upload them to airtable
    inventory_health_dataframe = inventory_health_download()
    inventory_health_diff_dataframe =  df_find_diff(airtable_dataframe, inventory_health_dataframe, False, False)
    airtable_update_columns(inventory_health_diff_dataframe, table)

"""Updates photos completed column in photo queue and the pair found column in image title base"""
def update_photo_completed():
    #Print message specifing what is being updated
    print('Photo Queue: Queue Updates')

    #Extract Queue Table from Photo Queue and convert it to a data to Pandas dataframe
    photo_queue_table = airtable_get_table('Photo Queue', 'Queue')
    photo_queue_airtable = airtable_download(photo_queue_table)

    #Extract Found Photos Table from Image Title and convert it to a data to Pandas dataframe
    image_title_table = airtable_get_table('Image Title', 'Found Photos')
    image_title_airtable = airtable_download(image_title_table)

    #Extract dataframes to be uploaded to airtable
    photo_queue_upload, image_title_upload = upc_found_dfs(image_title_airtable, photo_queue_airtable)
 
    #Upload them to airtable
    airtable_update_columns(photo_queue_upload, photo_queue_table)
    airtable_update_columns(image_title_upload, image_title_table)

"""Updates airtable based on the inputs into the funciton"""
def update_airtable(base = None, year = None, abbr = None):
    #Create dictonary of abbreviations to be bassed into dash and rank update functions
    abbreviations = ['CA', 'MX', 'US', 'NOAMT', 'FR', 'NL', 'PL', 'UK', 'GR', 'SP', 'SW', 'IT', 'EUT', 'EMEAT', 'SPK']

    #If no inputs entered run day to day updates on all bases
    if base == None and abbr == None and year == None:
        #Run day to day updates on Amazon, Amazon - SP and EMEA bases
        for abbr in abbreviations:
            update_dash(abbr)
            update_rank(abbr)

        #Run day to day updates on Ecomm base
        update_ecomm_ecom_styles()
        update_ecomm_amazon()

        #Run day to day updates on image bases
        update_photo_completed()

    #If just the year was entered run yearly updates on Amazon - US, Amazon - SP, and EMEA and regular update on Ecomm base
    if base == None and abbr == None:
        #Run full dash updates for that year on the Amazon - US, Amazon - SP and EMEA bases
        for abbr in abbreviations:
            update_dash_full(abbr, year)
            update_rank(abbr, year)

        #Run day to day updates on Ecomm base
        update_ecomm_ecom_styles()
        update_ecomm_amazon()

        #Run day to day updates on image bases
        update_photo_completed()

    #Othwise if the base entered is equal to Photo Queue
    elif base == 'Photo Queue':
        #Run day to day updates on image bases
        update_photo_completed()

    #Otherwise if the base entered is equal to Ecomm
    elif base == 'Ecomm':
        #Run day to day updates on Ecomm base
        update_ecomm_ecom_styles()
        update_ecomm_amazon()

    #Otherwise if the base entered is equal to Amazon - SP
    elif base == 'Saturday Park':
        #If no year was entered just run day to day updates on Amazon - SP base
        if year == None:
            update_dash(abbreviations[-1])
            update_rank(abbreviations[-1])
        
       #Otherwise if year was entered run full dash update for that year on Amazon - SP base (abbreviation is 'AMSP') 
        else:
            update_dash_full(abbreviations[-1], year)
            update_rank(abbreviations[-1], year)

    #Otherwise if the base entered is equal to Amazon
    elif base == 'NOAM':
        #Extract the NOAM abbreviations
        noam_abbreviations = abbreviations[:4]

        #If no year and no abbreviation was entered
        if year == None and abbr == None:
            #Run day to day updates for every country in the NOAM base
            for abbr in noam_abbreviations:
                update_dash(abbr)
                update_rank(abbr)

        #If year was entered but no abbreviation was entered
        elif abbr == None:
            #Run full dash updates for that year for every country in the NOAM base
            for abbr in noam_abbreviations:
                update_dash_full(abbr, year)
                update_rank(abbr, year)

        #If there was an abbreviation entered but no year was entered
        elif year == None:
            #Run day to day updates just on that abbreviation
            update_dash(abbr)
            update_rank(abbr)

        #Otherwise if all inputs were entered
        else:
            #Run full dash updates for that year for that country in the NOAM base
            update_dash_full(abbr, year)
            update_rank(abbr, year)
     
    #Otherwise if the base entered is equal to EMEA
    elif base == 'EMEA':
        #Extract the EMEA abbreviations
        emea_abbreviations = abbreviations[4:]

        #If no year and no abbreviation was entered
        if year == None and abbr == None:
            #Run day to day updates for every country in the EMEA base
            for abbr in emea_abbreviations:
                update_dash(abbr)
                update_rank(abbr)

        #If year was entered but no abbreviation was entered
        elif abbr == None:
            #Run full dash updates for that year for every country in the EMEA base
            for abbr in emea_abbreviations:
                update_dash_full(abbr, year)
                update_rank(abbr, year)

        #If there was an abbreviation entered but no year was entered
        elif year == None:
            #Run day to day updates just on that abbreviation
            update_dash(abbr)
            update_rank(abbr)

        #Otherwise if all inputs were entered
        else:
            #Run full dash updates for that year for that country in the EMEA base
            update_dash_full(abbr, year)
            update_rank(abbr, year)

"""
HOW TO USE UPDATE AIRTABLE FUNCTION:
    - No inputs -> Updates all airtable bases with day to day updates [update_airtable()]

    - Year input -> None followed by year will update Saturday Park, NOAM, and EMEA bases with yearly updates for the specified year and Ecomm and Photo Queue bases with day to day updates [update_airtable(None, 2022)]

    - Photo Queue Base -> Input 'Photo Queue' to run day to day updates on Photo Queue which must update Image Title Base [update_airtable('Photo Queue')]

    - Ecomm Base -> Input 'Ecomm' to run day to day updates on the Ecomm base [update_airtable('Ecomm')]

    - Saturday Park Base -> Input 'Saturday Park' as the base with or without the year
                    -> Inputting 'Saturday Park' will run day to day updates on the Saturday Park base [update_airtable('Saturday Park')]
                    -> Inputting 'Saturday Park' followed by the year will run a full yearly update on the Saturday Park base for the specified year [update_airtable('Saturday Park', 2022)]

    - NOAM Base -> Input 'NOAM' as the base with or without the year and with or without the abbreviation
                    -> Inputting 'NOAM' will run day to day updates (for all countries) on the NOAM base [update_airtable('NOAM')]
                    -> Inputting 'NOAM' followed by the year will run a full yearly update (for all countries) on the NOAM base for the specified year [update_airtable('NOAM', 2022)]
                    -> Inputting 'NOAM' followed by the year and a abbreviation will run a full yearly update (for that country) on the NOAM base for the specified year [update_airtable('NOAM', 2022, 'FR')]
                    -> Inputting 'NOAM' followed by None and a abbreviation will run day to day updates for that country on the NOAM base [update_airtable('NOAM', None, 'NL')]
                        -> Valid abbreviations are: 'CA', 'MX', 'US', 'NOAMT'

    - EMEA Base -> Input 'EMEA' as the base with or without the year and with or without the abbreviation
                    -> Inputting 'EMEA' will run day to day updates (for all countries) on the EMEA base [update_airtable('EMEA')]
                    -> Inputting 'EMEA' followed by the year will run a full yearly update (for all countries) on the EMEA base for the specified year [update_airtable('EMEA', 2022)]
                    -> Inputting 'EMEA' followed by the year and a abbreviation will run a full yearly update (for that country) on the EMEA base for the specified year [update_airtable('EMEA', 2022, 'FR')]
                    -> Inputting 'EMEA' followed by None and a abbreviation will run day to day updates for that country on the EMEA base [update_airtable('EMEA', None, 'NL')]
                        -> Valid abbreviations are: 'FR', 'NL', 'PL', 'UK', 'GR', 'SP', 'SW', 'IT', 'EUT', 'EMEAT'
"""

update_airtable()
