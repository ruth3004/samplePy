# fishPy
Another experiment structure for analysis pipeline
Flexible Infrastructure for Sample Handling in Python, Yay!

The fishPy structure should allow to create self objects that contain all experimental data (neurons, experimental protocol, etc.) that would allow to compare across other self objects. 
The scripts within this project should allow to create these structures and compare them across each other and within samples. 



Folder structure: 
    overview.csv /xml (file where self IDs are listed)
    
    preprocess (one self at a time)
        - create_struct
        - files_preprocessing
        - roi_prediction (rough_ROIing with Stardist/Cellpose)
        - motion_correction (BUJ, or cellpose, TrackMate solutions) 
        - extract_timetraces
        - cascade_timetraces (apply Cascade to infer neuron traces)

        image-anatomy functions
        - find_plane_in_stack
        - set_coordinates

        clem related functions
        - extract_lm_centroids
        - extract_em_cetroids
        - CLEM_mapping

        
    analysis 
        - mean_timetraces
        - heatmap
        - anatomy_maps
        - correlation_matrix
        - clustering (e.g. UMAP)

    batch (batch samples read on overview to preprocess, report or analyze)

    report (pdf and logs that summarize important steps)

    utils (functions used several times)