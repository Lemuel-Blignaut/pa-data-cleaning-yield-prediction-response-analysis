import os
import glob
import json
import geopandas as gpd
from rapidfuzz import process, fuzz


def files_to_dict(dir_path = None):
    # Strip quotes and whitespace that might come from the 'echo' command
    if not dir_path:
        folder_path = input("Enter the folder path: ").strip().strip('"').strip("'")
    else:
        folder_path = dir_path
        
    fp =  os.path.basename(folder_path)

    #Check if path actually exists before continuing
    if not os.path.exists(folder_path):
        print(f"Error: The directory '{fp}' does not exist.")
    else:
        # Use recursive glob
        # '**' matches any directory structure
        search_path = os.path.join(folder_path, '**', '*.shp')
        shp_files = glob.glob(search_path, recursive=True)
        fnames = [os.path.basename(f).replace('.shp', '') for f in shp_files]

        print(f"Found {len(shp_files)} shapefiles in {fp}...")
        print(f'List of shapefiles: {fnames}')

        dataframes = {}
        dict_entries = []
        for file in shp_files:
            # Get filename without extension for the dictionary key
            file_name = os.path.basename(file).replace('.shp', '')
            
            try:
                print(f"Importing file: {file_name}")
                dataframes[file_name] = gpd.read_file(file)
                dict_entries.append(file_name)
            except Exception as e:
                print(f"Failed to import {file_name}: {e}")

        print("Imported dataframes:", list(dataframes.keys()))
        
    return dataframes, fnames


def cast_to_crs(input_dict, crs):
    # cast the all of the dataframes in a dictionary to a common CRS,
    # defaulting to EPSG:32734 if none provided.
    if crs is None:
        crs = 'EPSG:32734'
        print(f"No CRS provided. Defaulting to {crs}")

    for key, df in input_dict.items():
        if df.crs is None:
            print(f"Warning: DataFrame {key} has no CRS defined. Setting CRS to {crs}.")
            input_dict[key] = df.set_crs(crs)
        else:
            input_dict[key] = df.to_crs(crs)
            print(f"Reprojected {key} to {crs}")
    
    return input_dict


def load_schema(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

  
def rename_columns_fuzzy(df, schema_dict, threshold=80, exclude_cols=None):
    """
    Parses a DataFrame and applies fuzzy matching to rename columns based on a schema.
    Returns a NEW DataFrame without altering the original.
    
    Parameters:
    - df: The input Pandas or GeoPandas DataFrame.
    - schema_dict: Dictionary mapping { 'standard_name': ['alias1', 'alias2'] }.
    - threshold: Minimum RapidFuzz score (0-100) to accept a match.
    - exclude_cols: List of column names to ignore (e.g., ['geometry']).
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Lowercase the excluded columns for safe, case-insensitive comparison
    exclude_cols_lower = [str(col).lower() for col in exclude_cols]
    
    # Track the best match for each standard name to prevent collisions
    # Format: { 'standard_name': ('original_column', score) }
    best_matches = {}
    
    for col in df.columns:
        # Skip columns we explicitly want to protect
        if str(col).lower() in exclude_cols_lower:
            continue
            
        best_score = 0
        best_standard_name = col
        
        # Compare the column against schema targets
        for std_name, aliases in schema_dict.items():
            targets = aliases + [std_name]
            match = process.extractOne(str(col), targets, scorer=fuzz.WRatio)
            
            if match:
                matched_str, score, _ = match
                if score > best_score and score >= threshold:
                    best_score = score
                    best_standard_name = std_name
                    
        # Collision handling: only keep the highest-scoring match for a standard name
        if best_standard_name != col:
            if best_standard_name in best_matches:
                existing_orig_col, existing_score = best_matches[best_standard_name]
                if best_score > existing_score:
                    best_matches[best_standard_name] = (col, best_score)
            else:
                best_matches[best_standard_name] = (col, best_score)
                
    # Build the final renaming dictionary
    rename_map = {orig_col: std_name for std_name, (orig_col, _) in best_matches.items()}
            
    # Return a new DataFrame with the renamed columns (inplace=False is the default)
    return df.rename(columns=rename_map)