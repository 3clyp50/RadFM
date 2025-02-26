# RadFM Medical Imaging Demo

This is a comprehensive Gradio-based demo for the RadFM (Radiology Foundation Model) that enables interactive visualization and analysis of both 2D and 3D medical images. The demo supports DICOM and NIFTI formats and includes a chat interface for querying the model about the images.

## Features

- Support for both 2D and 3D medical images (DICOM and NIFTI formats)
- Multiple view modes:
  - 2D/Single Slice
  - 3D/Multiplanar (Axial, Coronal, Sagittal)
  - 3D/Volume visualization
- Interactive slice navigation with sliders
- Image metadata display
- Chat interface for medical image analysis
- GPU acceleration support

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- RadFM model checkpoint and language files

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv radfm
   source radfm/bin/activate  # On Windows: radfm\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model files:
   - Get the RadFM checkpoint from [HuggingFace](https://huggingface.co/chaoyi-wu/RadFM)
   - Place `pytorch_model.bin` in the `RadFM/Quick_demo/` directory
   - Ensure the `Language_files` directory is present in `RadFM/Quick_demo/`

## Directory Structure

```
RadFM/
├── Quick_demo/
│   ├── pytorch_model.bin     # Download this from HuggingFace
│   ├── Language_files/       # Language model files
│   └── test.py              # Inference script
├── src/
│   └── Model/
│       └── RadFM/           # Model implementation files
└── gradio_demo/
    ├── app.py              # Gradio demo application
    ├── requirements.txt    # Dependencies
    └── README.md          # This file
```

## Running the Demo

1. Make sure you're in the RadFM root directory:
   ```bash
   cd RadFM
   ```

2. Run the Gradio demo:
   ```bash
   cd gradio_demo
   python app.py
   ```

3. Open your web browser and navigate to:
   - Local URL: http://127.0.0.1:7860
   - If running on a remote server, use the public URL provided in the terminal

## Using the Demo

1. **Loading the Model**:
   - Click the "Load Model" button to initialize RadFM
   - Wait for the confirmation message

2. **Uploading Images**:
   - Use the file upload button to select a DICOM (.dcm) or NIFTI (.nii, .nii.gz) file
   - The image will be automatically processed and displayed

3. **View Controls**:
   - Select view mode:
     - 2D/Single Slice: For basic 2D viewing
     - 3D/Multiplanar: For orthogonal plane visualization
     - 3D/Volume: For volume rendering
   - Use the view type selector to switch between:
     - Axial
     - Coronal
     - Sagittal
     - Multiplanar (shows all three views)

4. **Navigation**:
   - For 3D images, use the sliders to navigate through:
     - Axial slices
     - Coronal slices
     - Sagittal slices

5. **Image Analysis**:
   - Use the chat interface to ask questions about the current view
   - The model will analyze the visible portion of the image
   - Clear the chat history using the "Clear Chat" button

## Troubleshooting

1. **Import Errors**:
   - Ensure you're running the demo from the correct directory
   - Check that all required packages are installed
   - Verify the Python path includes the RadFM root directory

2. **Model Loading Issues**:
   - Confirm the model checkpoint is in the correct location
   - Check GPU memory availability
   - Ensure Language_files directory is present and accessible

3. **Image Loading Problems**:
   - Verify file format compatibility (DICOM or NIFTI)
   - Check file permissions
   - Ensure sufficient system memory

4. **Performance Issues**:
   - For large 3D volumes, consider reducing the input size
   - Close other GPU-intensive applications
   - Monitor system resources

## Known Limitations

- Maximum supported image dimensions: 512x512 pixels
- 3D volumes are sampled to 4 slices for model inference
- Chat responses are based on the current view only
- GPU memory requirements may be high for large volumes

## Citation

If you use this demo in your research, please cite the original RadFM paper:
```bibtex
@article{wu2023radfm,
  title={Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data},
  author={Wu, Chaoyi and others},
  journal={arXiv preprint arXiv:2308.02463},
  year={2023}
}
```

## Support

For issues and questions:
- Check the [RadFM GitHub repository](https://github.com/chaoyi-wu/RadFM)
- Contact the original authors for model-specific questions 