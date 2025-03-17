# import pandas as pd
# import numpy as np
# import requests
# import io

# def fetch_exoplanet_data():
#     """
#     Query the NASA Exoplanet Archive for confirmed planets with their 
#     parameters, host star information, and error values using the public API.
    
#     Returns:
#         pandas.DataFrame: DataFrame containing exoplanet and host star data with error values
#     """
#     print("Querying NASA Exoplanet Archive...")
    
#     # NASA Exoplanet Archive API URL for the Planetary Systems Composite Parameters table
#     # This contains the "best" values for each planet from all available references
#     base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
#     # Query parameters - requesting all columns which will include error values
#     params = {
#         'query': "select * from ps",  # Use the Planetary Systems (PS) table which has all planets
#         'format': 'csv'
#     }
    
#     # Make the request
#     print("Sending request to NASA Exoplanet Archive...")
#     response = requests.get(base_url, params=params)
    
#     if response.status_code == 200:
#         # Load the CSV data into a pandas DataFrame
#         exoplanets = pd.read_csv(io.StringIO(response.text))
#         print(f"Retrieved data for {len(exoplanets)} exoplanet records")
        
#         # List important columns that we'll need for our analysis, including error columns
#         # Define the main parameters first
#         main_params = [
#             # Planet parameters
#             'pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_radj', 'pl_bmassj', 
#             'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl', 'pl_eqt', 'pl_insol', 'pl_dens',
#             'discoverymethod', 'disc_year', 'pl_facility',
            
#             # Host star parameters
#             'st_spectype', 'st_teff', 'st_rad', 'st_mass', 'st_age', 'st_met', 'st_logg', 
#             'st_lum', 'st_dens', 'st_rotp',
            
#             # System parameters
#             'sy_snum', 'sy_pnum', 'sy_dist'
#         ]
        
#         # Generate a list of measurement parameters (exclude non-measurement fields)
#         measurement_params = [param for param in main_params 
#                              if param not in ['pl_name', 'hostname', 'discoverymethod', 
#                                              'disc_year', 'pl_facility', 'st_spectype']]
        
#         # Generate error columns for each measurement parameter
#         error_params = []
#         for param in measurement_params:
#             error_params.extend([f"{param}err1", f"{param}err2", f"{param}lim"])
        
#         # Combine main parameters with error parameters
#         important_cols = main_params + error_params
        
#         # Check which important columns exist in the data
#         available_cols = [col for col in important_cols if col in exoplanets.columns]
#         missing_cols = [col for col in important_cols if col not in exoplanets.columns]
        
#         if missing_cols:
#             print(f"Note: The following desired columns are not in the dataset: {missing_cols}")
            
#         return exoplanets
#     else:
#         print(f"Error retrieving data: Status code {response.status_code}")
#         print(response.text)
#         return None

# def alternative_query():
#     """
#     Alternative method to get exoplanet data using a direct CSV download.
#     This is useful if the TAP service isn't working.
#     """
#     print("Using alternative method to fetch exoplanet data...")
    
#     # NASA Exoplanet Archive PSCompPars CSV URL - requesting all columns for error values
#     csv_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv"
    
#     try:
#         # Read the CSV directly into a pandas DataFrame
#         exoplanets = pd.read_csv(csv_url)
#         print(f"Retrieved data for {len(exoplanets)} exoplanets")
#         return exoplanets
#     except Exception as e:
#         print(f"Error retrieving data via CSV: {e}")
#         return None

# def save_exoplanet_data(exoplanets, filename='nasa_exoplanet_data.csv'):
#     """
#     Save the retrieved exoplanet data to a CSV file
    
#     Args:
#         exoplanets (pandas.DataFrame): DataFrame of exoplanet data
#         filename (str): Output filename
#     """
#     if exoplanets is None or len(exoplanets) == 0:
#         print("No data to save")
#         return
        
#     exoplanets.to_csv(filename, index=False)
#     print(f"Data saved to {filename}")
    
#     # Also save a version with essential columns for causal analysis, including error values
#     # Define the main essential parameters
#     main_essential = [
#         'pl_name', 'hostname', 'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'pl_eqt', 
#         'pl_insol', 'discoverymethod', 'disc_year', 'st_age', 'st_teff', 'st_rad', 
#         'st_mass', 'st_met', 'st_logg', 'sy_snum', 'sy_pnum', 'sy_dist'
#     ]
    
#     # Define measurement parameters (excluding non-measurement fields)
#     measurement_essential = [param for param in main_essential 
#                            if param not in ['pl_name', 'hostname', 'discoverymethod', 'disc_year']]
    
#     # Add error columns for each measurement parameter
#     error_essential = []
#     for param in measurement_essential:
#         error_essential.extend([f"{param}_err1", f"{param}_err2", f"{param}_errlim"])
    
#     # Combine essential parameters with their error parameters
#     essential_cols = main_essential + error_essential
    
#     # Only include columns that exist in the dataframe
#     available_essential = [col for col in essential_cols if col in exoplanets.columns]
    
#     if available_essential:
#         exoplanets[available_essential].to_csv(
#             filename.replace('.csv', '_essential.csv'), 
#             index=False
#         )
#         print(f"Essential data saved to {filename.replace('.csv', '_essential.csv')}")
    
#     return

# def summarize_dataset(df):
#     """
#     Generate a summary of the dataset for initial examination
    
#     Args:
#         df (pandas.DataFrame): DataFrame of exoplanet data
#     """
#     if df is None or len(df) == 0:
#         print("No data to summarize")
#         return
        
#     print(f"\nDataset summary:")
#     print(f"Total exoplanet records: {len(df)}")
    
#     # Count unique planets
#     if 'pl_name' in df.columns:
#         print(f"Unique planets: {df['pl_name'].nunique()}")
    
#     # Discovery methods
#     if 'discoverymethod' in df.columns:
#         print("\nDiscovery methods:")
#         print(df['discoverymethod'].value_counts().head(10))
    
#     # Count planets with key parameters and their error values
#     key_params = ['pl_rade', 'st_age', 'st_mass', 'st_teff']
#     for param in key_params:
#         if param in df.columns:
#             param_count = df[param].notna().sum()
#             param_percent = param_count / len(df) * 100
#             print(f"Records with {param} data: {param_count} ({param_percent:.1f}%)")
            
#             # Check for error values if they exist
#             err1_col = f"{param}_err1"
#             err2_col = f"{param}_err2"
            
#             if err1_col in df.columns:
#                 err1_count = df[err1_col].notna().sum()
#                 err1_percent = err1_count / len(df) * 100
#                 print(f"  - with upper error ({err1_col}): {err1_count} ({err1_percent:.1f}%)")
                
#             if err2_col in df.columns:
#                 err2_count = df[err2_col].notna().sum()
#                 err2_percent = err2_count / len(df) * 100
#                 print(f"  - with lower error ({err2_col}): {err2_count} ({err2_percent:.1f}%)")
    
#     # Years of discovery
#     if 'disc_year' in df.columns:
#         print("\nDiscovery years range:")
#         print(f"From {df['disc_year'].min()} to {df['disc_year'].max()}")
    
#     # Count of columns with error information
#     error_cols = [col for col in df.columns if '_err1' in col or '_err2' in col or '_errlim' in col]
#     print(f"\nNumber of error-related columns: {len(error_cols)}")
#     print(f"Examples of error columns: {error_cols[:5]}")
    
#     return

# def analyze_error_availability(df):
#     """
#     Analyze the availability of error values for key parameters
    
#     Args:
#         df (pandas.DataFrame): DataFrame of exoplanet data
#     """
#     if df is None or len(df) == 0:
#         print("No data to analyze")
#         return
    
#     print("\nError Value Availability Analysis:")
    
#     # List of important parameters to check for errors
#     key_params = [
#         'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'pl_eqt', 'pl_insol',
#         'st_age', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg'
#     ]
    
#     results = []
    
#     for param in key_params:
#         if param in df.columns:
#             # Check if corresponding error columns exist
#             err1_col = f"{param}_err1"
#             err2_col = f"{param}_err2"
            
#             # Count values and errors
#             total_values = df[param].notna().sum()
            
#             if err1_col in df.columns:
#                 upper_errors = df[err1_col].notna().sum()
#                 upper_pct = (upper_errors / total_values * 100) if total_values > 0 else 0
#             else:
#                 upper_errors = 0
#                 upper_pct = 0
                
#             if err2_col in df.columns:
#                 lower_errors = df[err2_col].notna().sum()
#                 lower_pct = (lower_errors / total_values * 100) if total_values > 0 else 0
#             else:
#                 lower_errors = 0
#                 lower_pct = 0
                
#             results.append({
#                 'Parameter': param,
#                 'Total Values': total_values,
#                 'Upper Errors': upper_errors,
#                 'Upper Error %': upper_pct,
#                 'Lower Errors': lower_errors,
#                 'Lower Error %': lower_pct
#             })
    
#     # Convert to DataFrame for nice display
#     error_df = pd.DataFrame(results)
#     if not error_df.empty:
#         print(error_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    
#     return

# if __name__ == "__main__":
#     # Try the primary method first
#     exoplanets = fetch_exoplanet_data()
    
#     # If primary method fails, try the alternative
#     if exoplanets is None or len(exoplanets) == 0:
#         print("Primary query method failed, trying alternative method...")
#         exoplanets = alternative_query()
    
#     if exoplanets is not None and len(exoplanets) > 0:
#         # Provide a summary of the dataset
#         summarize_dataset(exoplanets)
        
#         # Analyze error value availability
#         analyze_error_availability(exoplanets)
        
#         # Save the data
#         save_exoplanet_data(exoplanets)
        
#         print("\nData retrieval and initial summary completed successfully.")
#     else:
#         print("Failed to retrieve exoplanet data from NASA Exoplanet Archive.")


import pandas as pd
import requests
import io

def fetch_exoplanet_data():
    """
    Query the NASA Exoplanet Archive for confirmed transiting planets with their 
    parameters, host star information, and error values using the public API.
    
    Returns:
        pandas.DataFrame: DataFrame containing exoplanet and host star data with error values
    """
    print("Querying NASA Exoplanet Archive for transiting planets...")
    
    # NASA Exoplanet Archive API URL for the Transiting Planets table
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Query parameters - requesting all columns from the transiting planets table
    query = "SELECT * FROM transiting"
    params = {
        'query': query,
        'format': 'csv'
    }
    
    # Make the request
    print("Sending request to NASA Exoplanet Archive...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        # Load the CSV data into a pandas DataFrame
        exoplanets = pd.read_csv(io.StringIO(response.text))
        print(f"Retrieved data for {len(exoplanets)} transiting exoplanet records")
        return exoplanets
    else:
        print(f"Error retrieving data: Status code {response.status_code}")
        print(response.text)
        return None

def save_exoplanet_data(exoplanets, filename='nasa_transiting_data.csv'):
    """
    Save the retrieved exoplanet data to a CSV file
    
    Args:
        exoplanets (pandas.DataFrame): DataFrame of exoplanet data
        filename (str): Output filename
    """
    if exoplanets is None or len(exoplanets) == 0:
        print("No data to save")
        return
        
    exoplanets.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return

def summarize_dataset(df):
    """
    Generate a summary of the dataset for initial examination
    
    Args:
        df (pandas.DataFrame): DataFrame of exoplanet data
    """
    if df is None or len(df) == 0:
        print("No data to summarize")
        return
        
    print(f"\nDataset summary:")
    print(f"Total exoplanet records: {len(df)}")
    
    # Count unique planets
    if 'pl_name' in df.columns:
        print(f"Unique planets: {df['pl_name'].nunique()}")
    
    # Discovery methods
    if 'discoverymethod' in df.columns:
        print("\nDiscovery methods:")
        print(df['discoverymethod'].value_counts().head(10))
    
    # Count planets with key parameters and their error values
    key_params = ['pl_rade', 'st_age', 'st_mass', 'st_teff']
    for param in key_params:
        if param in df.columns:
            param_count = df[param].notna().sum()
            param_percent = param_count / len(df) * 100
            print(f"Records with {param} data: {param_count} ({param_percent:.1f}%)")
            
            # Check for error values if they exist
            err1_col = f"{param}err1"
            err2_col = f"{param}err2"
            
            if err1_col in df.columns:
                err1_count = df[err1_col].notna().sum()
                err1_percent = err1_count / len(df) * 100
                print(f"  - with upper error ({err1_col}): {err1_count} ({err1_percent:.1f}%)")
                
            if err2_col in df.columns:
                err2_count = df[err2_col].notna().sum()
                err2_percent = err2_count / len(df) * 100
                print(f"  - with lower error ({err2_col}): {err2_count} ({err2_percent:.1f}%)")
    
    # Years of discovery
    if 'disc_year' in df.columns:
        print("\nDiscovery years range:")
        print(f"From {df['disc_year'].min()} to {df['disc_year'].max()}")
    
    # Count of columns with error information
    error_cols = [col for col in df.columns if 'err1' in col or 'err2' in col or 'lim' in col]
    print(f"\nNumber of error-related columns: {len(error_cols)}")
    print(f"Examples of error columns: {error_cols[:5]}")
    
    return

def analyze_error_availability(df):
    """
    Analyze the availability of error values for key parameters
    
    Args:
        df (pandas.DataFrame): DataFrame of exoplanet data
    """
    if df is None or len(df) == 0:
        print("No data to analyze")
        return
    
    print("\nError Value Availability Analysis:")
    
    # List of important parameters to check for errors
    key_params = [
        'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'pl_eqt', 'pl_insol',
        'st_age', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg'
    ]
    
    results = []
    
    for param in key_params:
        if param in df.columns:
            # Check if corresponding error columns exist
            err1_col = f"{param}err1"
            err2_col = f"{param}err2"
            
            # Count values and errors
            total_values = df[param].notna().sum()
            
            if err1_col in df.columns:
                upper_errors = df[err1_col].notna().sum()
                upper_pct = (upper_errors / total_values * 100) if total_values > 0 else 0
            else:
                upper_errors = 0
                upper_pct = 0
                
            if err2_col in df.columns:
                lower_errors = df[err2_col].notna().sum()
                lower_pct = (lower_errors / total_values * 100) if total_values > 0 else 0
            else:
                lower_errors = 0
                lower_pct = 0
                
            results.append({
                'Parameter': param,
                'Total Values': total_values,
                'Upper Errors': upper_errors,
                'Upper Error %': upper_pct,
                'Lower Errors': lower_errors,
                'Lower Error %': lower_pct
            })
    
    # Convert to DataFrame for nice display
    error_df = pd.DataFrame(results)
    if not error_df.empty:
        print(error_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    
    return

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    # Try the primary method first
    exoplanets = fetch_exoplanet_data()
    
    if exoplanets is not None and len(exoplanets) > 0:
        # Provide a summary of the dataset
        summarize_dataset(exoplanets)
        
        # Analyze error value availability
        analyze_error_availability(exoplanets)
        
        # Save the data
        save_exoplanet_data(exoplanets)
        
        print("\nData retrieval and initial summary completed successfully.")
    else:
        print("Failed to retrieve exoplanet data from NASA Exoplanet Archive.")
