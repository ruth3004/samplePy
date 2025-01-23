import os
import napari
import numpy as np
import tifffile
from skimage import measure
from napari_animation import Animation

parent_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd'

# Load data
em_stack = tifffile.imread(os.path.join(parent_folder, 'em_stack.tif'))
em_mask = tifffile.imread(os.path.join(parent_folder, 'em_mask.tif'))
em_centroids = np.load(os.path.join(parent_folder, 'em_centroids.npy'))

lm_stack = tifffile.imread(os.path.join(parent_folder, 'lm_stack.tif'))
lm_mask = tifffile.imread(os.path.join(parent_folder, 'lm_mask.tif'))
lm_centroids = np.load(os.path.join(parent_folder, 'lm_centroids.npy'))

em_warped_stack = tifffile.imread(os.path.join(parent_folder, 'em_stack_04_warped_fovLM.tif'))
em_warped_mask = tifffile.imread(os.path.join(parent_folder, 'em_stack_04_warped_fovLM_cp_masks.tif'))

# Calculate warped centroids
props = measure.regionprops(measure.label(em_warped_mask))
em_warped_centroids = np.array([prop.centroid for prop in props])

# Create viewer
viewer = napari.Viewer(ndisplay=2)

# Calculate the translation needed to align frame 752 of EM with frame 225 of LM
z_translation = 225 - 752

# Add layers with translations
em_stack_layer = viewer.add_image(em_stack, name='EM Stack', colormap='gray', visible=True, translate=[z_translation, 0, 0])
em_mask_layer = viewer.add_labels(em_mask, name='EM Mask', visible=False, translate=[z_translation, 0, 0])
em_centroids_layer = viewer.add_points(em_centroids, name='EM Centroids', size=5, face_color='magenta', visible=False, translate=[z_translation, 0, 0])

lm_stack_layer = viewer.add_image(lm_stack, name='LM Stack', colormap='gray', visible=False, translate=[0, 0, em_stack.shape[-1]])
lm_mask_layer = viewer.add_labels(lm_mask, name='LM Mask', visible=False, translate=[0, 0, em_stack.shape[-1]])
lm_centroids_layer = viewer.add_points(lm_centroids, name='LM Centroids', size=5, face_color='cyan', visible=False, translate=[0, 0, em_stack.shape[-1]])

em_warped_stack_layer = viewer.add_image(em_warped_stack, name='EM Warped Stack', colormap='gray', visible=False, translate=[0, 0, em_stack.shape[-1]])
em_warped_mask_layer = viewer.add_labels(em_warped_mask, name='EM Warped Mask', visible=False, translate=[0, 0, em_stack.shape[-1]])
em_warped_centroids_layer = viewer.add_points(em_warped_centroids, name='EM Warped Centroids', size=5, face_color='yellow', visible=False, translate=[0, 0, em_stack.shape[-1]])

# Create animation
animation = Animation(viewer)

# 1. EM stack scrolling
animation.capture_keyframe()
for i in range(0, em_stack.shape[0], 20):  # Increased step size
    viewer.dims.set_point(0, i)
    animation.capture_keyframe()

# 2. EM stack + mask scrolling + EM centroids
em_mask_layer.visible = True
em_centroids_layer.visible = True
animation.capture_keyframe()
for i in range(em_stack.shape[0]-1, 0, -20):  # Increased step size
    viewer.dims.set_point(0, i)
    animation.capture_keyframe()

# 3. LM stack at the comparison frame + first centroid
em_stack_layer.visible = False
em_mask_layer.visible = False
em_centroids_layer.visible = False
lm_stack_layer.visible = True
lm_centroids_layer.visible = True
viewer.dims.set_point(0, 225)  # Set to comparison frame
animation.capture_keyframe(steps=30)  # Reduced steps

# 4. Warp EM centroid landmarks to LM centroid landmarks
em_warped_centroids_layer.visible = True
animation.capture_keyframe(steps=30)  # Reduced steps

# 5. Scrolling EM warped + LM stacks overlaying
em_warped_stack_layer.visible = True
em_warped_stack_layer.opacity = 0.5
lm_stack_layer.opacity = 0.5
for i in range(0, lm_stack.shape[0], 20):  # Increased step size
    viewer.dims.set_point(0, i)
    animation.capture_keyframe()

# Adjust viewer size to be divisible by 16
viewer.window.resize(896, 880)

# Save animation
animation.animate('clem_alignment_sequence.mp4', fps=30)  # Increased fps

napari.run()