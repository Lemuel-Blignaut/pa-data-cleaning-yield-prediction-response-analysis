import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shapely as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree


def calculate_dominant_heading(points_df, plots = False):
    """
    Calculates the dominant driving angle of the harvester by analyzing 
    the trajectory between consecutive GPS points.
    
    Returns the angle in degrees (0-180).
    """
    print("Calculating dominant harvester heading...")
    
    # Extract coordinates
    coords = np.column_stack((points_df.geometry.x, points_df.geometry.y))
    
    # Calculate the difference (dx, dy) between consecutive points
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    
    # Calculate the angle of travel for each step in degrees
    angles = np.degrees(np.arctan2(dy, dx))
    
    # Harvesters drive back and forth (e.g., 45° then 225°). 
    # We use modulo 180 to fold them into the same orientation plane.
    orientation = angles % 180
    
    # Drop NaNs or infinite values (where the harvester didn't move)
    orientation = orientation[~np.isnan(orientation)]
    
    # Create a histogram to find the most common angle (1-degree bins)
    hist, bin_edges = np.histogram(orientation, bins=180, range=(0, 180))
    dominant_angle = bin_edges[np.argmax(hist)]
    
    if plots:
        plt.figure(figsize=(6, 4))
        plt.hist(orientation, bins=180, range=(0, 180), color='skyblue', edgecolor='black')
        plt.axvline(x=dominant_angle, color='red', linestyle='--', label=f'Dominant Heading: {dominant_angle:.1f} degrees')
        plt.title('Distribution of Harvester Headings')
        plt.xlabel('Heading Angle (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

        
    print(f"Dominant heading found: {dominant_angle:.1f}°")
    return dominant_angle


def create_spatial_grid(input_df, cell_size = None, width_col = None, length_col = None, shape = 'square', angle = 0, plots = False,value_col = None, boundary_df = None, buffer_radius = None):
    """_summary_

    Args:
        input_df (_type_): _description_
        cell_size (_type_, optional): _description_. Defaults to None.
        width_col (_type_, optional): _description_. Defaults to None.
        length_col (_type_, optional): _description_. Defaults to None.
        shape (str, optional): _description_. Defaults to 'square'.
        angle (int, optional): _description_. Defaults to 0.
        plots (bool, optional): _description_. Defaults to False.
        value_col (_type_, optional): _description_. Defaults to None.
        boundary_df (_type_, optional): _description_. Defaults to None.
        buffer_radius (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """
   
    """
    Generates a square fishnet grid, with an optional rotation angle 
    to align with harvester passes.
    
    Args:
        input_df (GeoDataFrame): The input data to bound the grid.
        cell_size (float): Size of the grid cells in meters.
        angle (float): Rotation angle in degrees.
    """
    
    try:
        if shape == 'square' and cell_size is not None:
            print(f"Creating {cell_size}m square spatial grid (Rotation: {angle:.1f} degrees)")
            width = cell_size
            length = cell_size
        elif shape == 'rectangle':
            width = np.round(input_df[width_col].mean(),2)
            length = np.round(input_df[length_col].mean(),2)
            print(f"Creating a {width} x {length} m spatial grid (Rotation: {angle:.1f} degrees)")
    except Exception as e:
        print(f"{e}") 
    
    
    #  If an angle is provided, temporarily counter-rotate the input data 
    # to find the absolute minimum bounding box needed to cover the rotated shape.
    if angle != 0:
        pivot = input_df.unary_union.centroid
        temp_geom = input_df.geometry.rotate(-angle, origin=pivot)
        minx, miny, maxx, maxy = temp_geom.total_bounds
    else:
        minx, miny, maxx, maxy = input_df.total_bounds
 
    # Create the coordinate arrays
    x_coords = np.arange(minx-2*length, maxx+2*length, length)
    # NB the y-coords here and in crete box has been manipulated, this more or less replicates harvester
    # footprint for Voerstoor. not generally applicable yet
    y_coords = np.arange(miny-2*width, maxy+2*width, width)
    
    polygons = []
    # Build square polygons for each cell
    for x in x_coords:
        for y in y_coords:
            polygons.append(sp.box(x, y, x + length, y + width))
            
    # Create the GeoDataFrame and give each cell a unique ID
    grid_gdf = gpd.GeoDataFrame(
        {'grid_id': range(len(polygons))}, 
        geometry=polygons, 
        crs=input_df.crs
    )
    
    if angle != 0:
        grid_gdf.geometry = grid_gdf.geometry.rotate(angle, origin=pivot)
    
    if boundary_df is not None:
        b = boundary_df.union_all().buffer(buffer_radius)
        grid_gdf = gpd.clip(grid_gdf, b)
    
    if plots:
        fig,ax = plt.subplots(figsize=(6,4))
        grid_gdf.plot(ax=ax, facecolor='none', alpha = 0.5, edgecolor='black', linewidth=0.1)
        if input_df is not None:
            input_df.plot(column = value_col,cmap = 'RdYlGn', ax=ax, legend=True, markersize = 3,
                          legend_kwds = {'label': 'Heading (degrees)'})
        if boundary_df is not None:
            boundary_df.plot(ax=ax, facecolor='none', edgecolor='red')
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.axis('equal')
        plt.title(f'Regular polygonal grid aligned to primary harvester heading')
        plt.tight_layout()
        plt.show()
    
    print(f"Grid created with {len(grid_gdf)} cells.")
    return grid_gdf, width


def aggregate_points_to_grid(points_df, grid_gdf, value_col, is_rate=True):
    """
    Aggregates point values to a grid. 
    Handles both intensive variables (rates like t/ha) and extensive variables (mass like kg).
    """
    print(f"Aggregating '{value_col}' values to grid...")
    
    # Ensure CRS matches
    if points_df.crs != grid_gdf.crs:
        print(f"CRS mismatch. Casting points_df to grid_gdf CRS ({grid_gdf.crs})")
        points_df = points_df.to_crs(grid_gdf.crs)
        
    # Resetting index ensures no accidental data loss if original indices were non-unique
    points_df = points_df.reset_index(drop=True)
    
    # 'within' prevents a point on a boundary line from being assigned to multiple cells
    merged = gpd.sjoin(points_df, grid_gdf, how='inner', distance = 0.75, predicate='dwithin')
    print(f"Columns in merged DF: {merged.columns.tolist()}")
    # Using groupby is vastly superior to dissolve for pointwise operations
    if is_rate:
        # If dealing with t/ha, we want the mean rate per cell
        agg_series = merged.groupby('index_right')[value_col].mean()
    else:
        # If dealing with absolute mass (e.g. kg), we want the sum per cell
        agg_series = merged.groupby('index_right')[value_col].sum()
        
    # Safe assignment using the grid's native index
    filled_grid_gdf = grid_gdf.copy()
    filled_grid_gdf[value_col] = filled_grid_gdf.index.map(agg_series)
    
    # Final Verification
    print(f"Original point count: {len(points_df)}")
    print(f"Joined point count: {len(merged)}") 
    
    if is_rate:
        # We must calculate cell areas to convert rates back to masses for comparison.
        # Dropping NaNs ensures we only calculate mass for grid cells that actually received data.
        valid_grid = filled_grid_gdf.dropna(subset=[value_col])
        cell_areas = valid_grid.geometry.area
        
        # 1. Gridded total mass = Sum of (Cell Rate * Cell Area)
        grid_total_mass = (valid_grid[value_col] * cell_areas).sum()
        
        # 2. Original total mass estimate = Overall Mean Point Rate * Total Valid Grid Area
        # (This approximates the raw mass, assuming points were evenly distributed)
        orig_mean_rate = merged[value_col].mean()
        total_intersected_area = cell_areas.sum()
        orig_total_mass_est = orig_mean_rate * total_intersected_area
        
        print(f"Original Total Mass (Estimated): {orig_total_mass_est:.2f}")
        print(f"Gridded Total Mass (Area-Weighted): {grid_total_mass:.2f}")
        
        if orig_total_mass_est > 0:
            diff = abs(orig_total_mass_est - grid_total_mass) / orig_total_mass_est * 100
            print(f"Difference: {diff:.2f}%")
    else:
        orig_total = round(points_df[value_col].sum(), 2)
        grid_total = round(filled_grid_gdf[value_col].sum(), 2)
        print(f"Original Total Mass: {orig_total}")
        print(f"Gridded Total Mass: {grid_total}")
        print(f"Success? {orig_total == grid_total}")
        
    return filled_grid_gdf


def assign_polygon_to_grid(polygon_df, grid_gdf, value_col):
    """
    Overlays polygons onto the grid and calculates area-weighted values.
    """
    print(f"Assigning polygon '{value_col}' to grid via area-weighting...")
    
    # ensure CRS matches
    if polygon_df.crs != grid_gdf.crs:
        polygon_df = polygon_df.to_crs(grid_gdf.crs)
        print(f"CRS don't match. Casting {polygon_df} to {grid_gdf} CRS")
    
    # Create a copy so we don't directly modify the input GeoDataFrame
    final_grid = grid_gdf.copy()
    
    # WEIGHTED POLYGON ASSIGNMENT
            
    # 1. OVERLAY: This actually cuts the polygons to fit inside the grid cells
    # It keeps attributes from both DataFrames
    overlay_df = gpd.overlay(final_grid, polygon_df, how='intersection',keep_geom_type=False)
    
    # 2. Calculate the area of the overlapped fragment
    overlay_df['overlap_area'] = overlay_df.geometry.area
    
    # 3. Get the area of the grid cells.
    # We map by 'grid_id' rather than using .iloc[0] to ensure accuracy 
    # if grid cells have slight area variations across projection spaces.
    cell_areas = final_grid.set_index('grid_id').geometry.area
    overlay_df['cell_area'] = overlay_df['grid_id'].map(cell_areas)
    
    # 4. Calculate the weight (% of the cell covered)
    overlay_df['weight'] = overlay_df['overlap_area'] / overlay_df['cell_area']
    
    # 5. Apply the weight to the value
    overlay_df['weighted_value'] = overlay_df[value_col] * overlay_df['weight']
    
    # 6. Sum the weighted values per grid cell
    agg_series = overlay_df.groupby('grid_id')['weighted_value'].sum()

    # 7. Map back to the master grid. 
    # Using .map() instead of .merge() prevents pandas from resetting the index 
    # and safely preserves final_grid as a GeoDataFrame.
    final_grid[value_col + "_weighted"] = final_grid['grid_id'].map(agg_series)

    # UNWEIGHTED POLYGON ASSIGNMENT
    
    # 'inner' ensures we only look at points that actually hit the grid
    merged = gpd.sjoin(polygon_df, final_grid, how='inner', predicate='intersects')
  
    # Using .groupby() is vastly more computationally efficient than .dissolve() 
    # here because we only require the mathematical mean, not the merged geometries.
    # 'index_right' natively holds the exact index values of final_grid.
    yield_per_cell = merged.groupby("index_right")[value_col].mean()

    # Direct assignment. Pandas will now align these perfectly because 
    # yield_per_cell.index maps flawlessly to final_grid.index.
    final_grid[value_col] = yield_per_cell
    
    return final_grid


def align_to_grid(master_grid, gdf, col, method='nearest', search_radius=10, p=2, k=5):
    """
    Align a GeoDataFrame's column to the master grid geometry using spatial query.
    Works seamlessly with both POINT and POLYGON master grids.
    
    Parameters:
    - master_grid: GeoDataFrame of the target points/polygons grid.
    - gdf: GeoDataFrame of the source points/polygons.
    - col: Column name in gdf to align.
    - method: 'nearest' (closest point value) or 'idw' (Inverse Distance Weighting).
    - search_radius: Maximum distance (meters) to search for points.
                     If None, no limit is enforced (continuous interpolation).
    - p: IDW power parameter.
    - k: Number of nearest neighbors to query for IDW.
    """
    # Robust coordinate extraction for target grid (handles polygons via centroids)
    if master_grid.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
        centroids = master_grid.geometry.centroid
        grid_coords = np.column_stack((centroids.x, centroids.y))
    else:
        grid_coords = np.column_stack((master_grid.geometry.x, master_grid.geometry.y))
    
    # CRITICAL FIX 1: Filter out rows where the target column is NaN in the SOURCE data
    # This prevents NaN values from contaminating neighbors during interpolation.
    clean_gdf = gdf.dropna(subset=[col])
    if len(clean_gdf) == 0:
        return np.full(len(master_grid), np.nan)
        
    # Robust coordinate extraction for source layer (handles source polygons via centroids)
    if clean_gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
        src_centroids = clean_gdf.geometry.centroid
        gdf_coords = np.column_stack((src_centroids.x, src_centroids.y))
    else:
        gdf_coords = np.column_stack((clean_gdf.geometry.x, clean_gdf.geometry.y))
    
    # Build KDTree using only valid data points
    tree = KDTree(gdf_coords)
    
    if method == 'nearest':
        dist, ind = tree.query(grid_coords, k=1)
        dist = dist.ravel()
        ind = ind.ravel()
        
        # Extract values
        aligned_vals = clean_gdf[col].iloc[ind].values
        
        # Apply search radius mask (if search_radius is provided)
        if search_radius is not None:
            aligned_vals = np.where(dist <= search_radius, aligned_vals, np.nan)
            
        return aligned_vals
        
    elif method == 'idw':
        # Adjust k if we have fewer valid points than requested k
        actual_k = min(k, len(clean_gdf))
        dist, ind = tree.query(grid_coords, k=actual_k)
        
        # Calculate IDW weights (with a small epsilon to prevent division by zero)
        eps = 1e-12
        weights = 1.0 / (dist ** p + eps)
        
        # Handle exact overlaps (distance is 0) to avoid numeric instability
        exact_matches = dist < 1e-5
        if np.any(exact_matches):
            row_has_exact = np.any(exact_matches, axis=1)
            for i in range(len(grid_coords)):
                if row_has_exact[i]:
                    weights[i] = exact_matches[i].astype(float)
                    
        # Normalize weights
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weights_sum
        
        # Retrieve neighbor values
        neighbor_vals = clean_gdf[col].iloc[ind.ravel()].values.reshape(ind.shape)
        
        # Compute weighted average
        aligned_vals = np.sum(neighbor_vals * weights, axis=1)
        
        # Apply search radius mask (based on the NEAREST neighbor in the query)
        if search_radius is not None:
            aligned_vals = np.where(dist[:, 0] <= search_radius, aligned_vals, np.nan)
            
        return aligned_vals
    
    else:
        raise ValueError("Method must be 'nearest' or 'idw'")
    
    
def add_spatial_lag(grid_gdf, value_col, radius=50.0):
    """
    Adds a new column 'value_col_lag' representing the mean value 
    of neighbors within a specified radius (in meters).
    """
    # 1. Ensure we are in a metric CRS
    if grid_gdf.crs.is_geographic:
        raise ValueError("Grid must be in a projected (metric) CRS to calculate radius-based lags.")
    
    # 2. Extract coordinates
    coords = np.column_stack((grid_gdf.geometry.centroid.x, grid_gdf.geometry.centroid.y))
    tree = KDTree(coords)
    
    # 3. Query the tree for neighbors within radius
    # indices is a list of arrays of neighbor indices for each point
    indices = tree.query_radius(coords, r=radius)
    
    # 4. Calculate mean of neighbors
    lag_values = []
    for i, neighbors in enumerate(indices):
        # Calculate mean of the value_col for all neighbors
        # We exclude the point itself if we want a "pure" spatial lag
        mask = neighbors[neighbors != i]
        if len(mask) > 0:
            lag_values.append(grid_gdf.iloc[mask][value_col].mean())
        else:
            lag_values.append(grid_gdf.iloc[i][value_col]) # Fallback to own value
            
    grid_gdf[f"{value_col}_spatial_lag"] = lag_values
    return grid_gdf


def kmeans_clustering(grid_gdf, cols, name, n_clusters=3, auto_k=False, max_k=10, n_init=10, random_state=42, plots=False):
    """
    Applies KMeans clustering to the specified columns in the grid GeoDataFrame.
    Automatically finds the optimal cluster number using Silhouette Score if auto_k=True.
    
    Args:
        grid_gdf (GeoDataFrame): The input grid with values to cluster.
        cols (list): List of column names to use for clustering.
        name (str): Name for the new cluster column.
        n_clusters (int): Number of clusters to form (used if auto_k=False).
        auto_k (bool): If True, automatically calculates optimal k using Silhouette Score.
        max_k (int): Maximum number of clusters to check if auto_k=True.
        n_init (int): Number of times the k-means algorithm will be run with different centroid seeds.
        random_state (int): Determines random number generation for centroid initialization.
        plots (bool): Whether to plot the clusters on a map (and silhouette curve if auto_k=True).
        
    Returns:
        df (GeoDataFrame): A GeoDataFrame with an added 'cluster' column indicating cluster membership.
    """
    df = grid_gdf.copy()
    
    # 1. Select features for clustering
    X = df[cols].values
    
    # 2. Determine optimal k if auto_k is enabled
    if auto_k:
        best_score = -1
        best_k = 2
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X))) # ensure we don't exceed sample size
        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=random_state)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        n_clusters = best_k
        print(f"Optimal number of clusters found: {n_clusters} (Silhouette Score: {best_score:.4f})")
        
        # Optional: Plot the Silhouette Score curve
        if plots:
            plt.figure(figsize=(6, 4))
            plt.plot(list(k_range), silhouette_scores, marker='o', color='purple')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.axvline(x=best_k, color='red', linestyle='--', label=f'Optimal k={best_k}')
            plt.legend()
            plt.show()

    # 3. Configure and run KMeans with final n_clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, random_state=random_state)
    kmeans.fit(X)

    # 4. Extract results
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    if len(labels) == len(df):
        df[name] = labels

    # Plot final clusters
    if plots:
        plt.figure(figsize=(6, 4))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
        plt.title(f"K-Means Clustering (k={n_clusters})")
        plt.legend()
        plt.show()
        
    return df


def impute_spatial_mean(gdf, column_name, method="rook", k=4):
    """Imputes missing values in a GeoDataFrame column using the mean of its neighbors.

    Parameters:
    - gdf: GeoDataFrame
    - column_name: String, the column with missing data (NaNs)
    - method: 'rook'/'queen' (for polygons sharing borders) or 'knn' (for points/distance)
    - k: Number of neighbors to use if method='knn'
    """
    # 1. Build the spatial weights matrix based on chosen neighborhood type
    if method == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    elif method == "rook":
        w = libpysal.weights.Rook.from_dataframe(gdf, use_index=False)
    elif method == "knn":
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k, use_index=False) #need to adapt this still based on this https://pysal.org/libpysal/stable/generated/libpysal.weights.KNN.html
    else:
        raise ValueError("Method must be 'queen', 'rook', or 'knn'")

    # Create a copy of the target column to work on
    imputed_series = gdf[column_name].copy()

    # 2. Iterate through rows that have missing values
    missing_indices = gdf[gdf[column_name].isna()].index

    for idx in missing_indices:
        # Get the integer positions or IDs of neighbors
        # w.neighbors maps each row index to a list of neighbor indices
        neighbors_idx = w.neighbors[idx]

        if len(neighbors_idx) > 0:
            # Extract neighbor values
            neighbor_values = gdf.loc[neighbors_idx, column_name]

            # Calculate mean, safely ignoring any neighboring NaNs
            neighbor_mean = neighbor_values.mean()

            # Fill the missing spot if we found a valid mean
            if not np.isnan(neighbor_mean):
                imputed_series.loc[idx] = neighbor_mean

    return imputed_series




