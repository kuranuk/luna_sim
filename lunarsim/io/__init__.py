"""File I/O (PNG/NPY rasters, point clouds and trajectories in later phases).

Never uses pickle: ``np.load`` is always called with ``allow_pickle=False``.
"""
