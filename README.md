## samplePy

Simple Analysis for Multimodal Pipelines of Light and Electron imaging (samplePy)

A pipeline for analyzing and integrating light microscopy (LM) and electron microscopy (EM) data in neuroscience experiments.
samplePy is designed to process, analyze, and correlate multi-modal imaging data from neuronal samples. It provides a structured workflow from raw data input to advanced analysis and visualization.

## Pipeline Structure

The pipeline is organized into several key stages:


1. **LM Data Processing**
   - Load experiments and create samples
   - Register LM trials and stacks
   - Segment cells and extract traces
   - Normalize, deconvolve, and correlate traces

2. **LM Stack Processing**
   - Preprocess LM stacks
   - Register to reference stack
   - Segment with EM-warped masks
   - Extract markers from channels

3. **EM Data Processing**
   - Segment cells using SOFIMA alignment and Cellpose
   - Segment glomeruli
   - Find landmarks with BigWarp 
   - Register to LM stack and trials

4. **CLEM (Correlative Light and Electron Microscopy)**
   - Register centroids using LUT (Look-Up Table)

5. **Analysis**
   - Functional-structural analysis (details to be implemented)

## Key Features

- Integrated processing of LM trials, LM stacks, and EM stacks
- Advanced cell segmentation and trace extraction
- Cross-modality registration and correlation
- Flexible pipeline with modular steps

## Usage



