import numpy as np
import geopandas as gpd
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed


def terrain(x,y,z, crit, s):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        crit (_type_): _description_
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    points = np.column_stack((x, y, z))
    tree = KDTree(points[:, :2])
    n_jobs = -1 # Use all cores; or set to something like 4

    # QUERY THE TREE ON THE MAIN THREAD
    if crit == 'dist':
        # This returns a list of arrays (the indices of neighbors for each point)
        all_neighbor_indices = tree.query_radius(points[:, :2], r=s)
    elif crit == 'neighbours':
        _, all_neighbor_indices = tree.query(points[:, :2], k=s)

    # NOW RUN PARALLEL
    # We only pass the specific subset of points needed for that specific calculation
    results = Parallel(n_jobs=n_jobs, batch_size=500)(
        delayed(fit_plane_from_subset)(points[idx]) for idx in all_neighbor_indices
    )

    # Unpack into columns
    results_df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs='EPSG:32734')
    results_df['elevation'] = z
    results_df['slope_deg'] = [r[0] for r in results]
    results_df['aspect_deg'] = [r[1] for r in results]


    return results_df


def fit_plane_from_subset(neighbor_points):
    # 1. Filter out any hidden NaNs in this specific neighborhood
    # This prevents the "Poisoned Apple" effect
    valid_points = neighbor_points[~np.isnan(neighbor_points).any(axis=1)]

    # 2. We mathematically need at least 3 valid points to fit a plane.
    # If the radius hits a sparse edge zone, return NaN safely instead of crashing.
    if len(valid_points) < 3:
        return np.nan, np.nan

    # 3. Center the math on the local centroid. 
    # This completely bypasses the unsorted KDTree bug and stabilizes the SVD solver.
    centroid = np.mean(valid_points, axis=0)
    xi, yi, zi = centroid[0], centroid[1], centroid[2]
    
    xk, yk, zk = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
    
    # 4. Build the matrix and solve
    A = np.column_stack((xk - xi, yk - yi, np.ones(len(valid_points))))
    coeffs, _, _, _ = np.linalg.lstsq(A, zk - zi, rcond=None)
    a, b = coeffs[0], coeffs[1]

    # 5. Calculate Slope and Aspect
    slope = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))
    aspect = np.degrees(np.arctan2(-a, -b)) % 360

    return slope, aspect


