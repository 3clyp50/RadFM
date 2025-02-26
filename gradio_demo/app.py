import os
import sys
import torch
import numpy as np
import gradio as gr
from pathlib import Path
import tempfile
import pydicom
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import make_grid

# Add the parent directory to sys.path to make the imports work
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# Import RadFM model components
from src.Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
# Import from Quick_demo/test.py using explicit relative path
sys.path.insert(0, str(root_dir / 'Quick_demo'))
from test import get_tokenizer, combine_and_preprocess

# Set device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and tokenizer
global_model = None
global_tokenizer = None
global_image_padding_tokens = None
checkpoint_path = os.path.join(root_dir, "Quick_demo", "pytorch_model.bin")
lang_model_path = os.path.join(root_dir, "Quick_demo", "Language_files")

def load_model_and_tokenizer():
    """Load the RadFM model and tokenizer."""
    global global_model, global_tokenizer, global_image_padding_tokens
    
    print("Loading tokenizer...")
    global_tokenizer, global_image_padding_tokens = get_tokenizer(lang_model_path)
    print("Tokenizer loaded successfully!")
    
    print("Loading model...")
    global_model = MultiLLaMAForCausalLM(lang_model_path=lang_model_path)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        global_model.load_state_dict(ckpt)
        global_model = global_model.to(device)
        global_model.eval()
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint from https://huggingface.co/chaoyi-wu/RadFM")
        print("and place it in the Quick_demo directory.")
    
    return "Model and tokenizer loaded successfully!"

# Helper functions for image processing
def normalize_array(array, min_val=None, max_val=None):
    """Normalize array to [0, 1] range."""
    if min_val is None:
        min_val = array.min()
    if max_val is None:
        max_val = array.max()
    
    return (array - min_val) / (max_val - min_val + 1e-10)

def read_dicom(file_path):
    """Read DICOM file and return pixel array."""
    try:
        dcm = pydicom.dcmread(file_path)
        pixel_array = dcm.pixel_array
        
        # Extract DICOM metadata
        metadata = {
            "Patient ID": dcm.get("PatientID", "N/A"),
            "Patient Name": str(dcm.get("PatientName", "N/A")),
            "Modality": dcm.get("Modality", "N/A"),
            "Study Date": dcm.get("StudyDate", "N/A"),
            "Slice Location": dcm.get("SliceLocation", "N/A"),
            "Image Position": dcm.get("ImagePosition", "N/A"),
        }
        
        # Normalize to 0-1 range
        normalized_array = normalize_array(pixel_array)
        
        # Determine if it's 2D or 3D
        if len(normalized_array.shape) > 2:  # 3D or multi-frame DICOM
            is_3d = True
            depth = normalized_array.shape[0]
        else:  # 2D DICOM
            is_3d = False
            depth = 1
            normalized_array = normalized_array[np.newaxis, ...]
        
        return normalized_array, metadata, is_3d, depth
    
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None, None, False, 0

def read_nifti(file_path):
    """Read NIFTI file and return pixel array."""
    try:
        img = nib.load(file_path)
        pixel_array = img.get_fdata()
        
        # Extract NIFTI metadata
        metadata = {
            "Dimensions": f"{img.shape}",
            "Voxel Size": f"{img.header.get_zooms()}",
            "Datatype": f"{img.header.get_data_dtype()}",
            "Orientation": f"{nib.aff2axcodes(img.affine)}",
        }
        
        # Normalize to 0-1 range
        normalized_array = normalize_array(pixel_array)
        
        # Determine if it's 2D or 3D
        if len(normalized_array.shape) > 2:  # 3D NIFTI
            is_3d = True
            # Handle different orientations
            if normalized_array.shape[2] == 1:  # If third dimension is 1, it's essentially 2D
                normalized_array = np.transpose(normalized_array, (2, 0, 1))
                is_3d = False
            else:
                normalized_array = np.transpose(normalized_array, (2, 0, 1))  # Put depth as first dimension
            depth = normalized_array.shape[0]
        else:  # 2D NIFTI (rare)
            is_3d = False
            depth = 1
            normalized_array = normalized_array[np.newaxis, ...]
        
        return normalized_array, metadata, is_3d, depth
    
    except Exception as e:
        print(f"Error reading NIFTI file: {e}")
        return None, None, False, 0

def slice_to_pil(slice_data):
    """Convert a 2D slice to PIL image."""
    # Apply colormap for better visualization
    colored_slice = cm.gray(slice_data)
    colored_slice = (colored_slice * 255).astype(np.uint8)
    
    # Convert to PIL Image
    slice_image = Image.fromarray(colored_slice)
    return slice_image

def preprocess_for_model(image_numpy):
    """Preprocess image for RadFM model input."""
    # Convert to torch tensor
    if len(image_numpy.shape) == 3:  # 3D volume
        # Select a subset of slices (up to 4) for the model
        depth = image_numpy.shape[0]
        stride = max(1, depth // 4)
        selected_slices = image_numpy[::stride][:4]
        
        # Pad if needed
        if selected_slices.shape[0] < 4:
            padding = np.zeros((4 - selected_slices.shape[0], *selected_slices.shape[1:]))
            selected_slices = np.concatenate([selected_slices, padding], axis=0)
            
        # Convert to tensor with shape [C, H, W, D]
        image_tensor = torch.from_numpy(selected_slices).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
    else:  # 2D image
        image_tensor = torch.from_numpy(image_numpy).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(-1)  # Add channel and depth dimensions
    
    # Resize to target dimensions
    target_H = 512
    target_W = 512
    target_D = 4
    image_tensor = F.interpolate(image_tensor, size=(target_H, target_W, target_D))
    
    return image_tensor

def run_inference(image_array, prompt):
    """Run RadFM model inference on the given image and prompt."""
    global global_model, global_tokenizer, global_image_padding_tokens
    
    if global_model is None or global_tokenizer is None:
        return "Model not loaded. Please load the model first."
    
    # Preprocess the image
    image_tensor = preprocess_for_model(image_array)
    
    # Combine text and images for model input
    question = [prompt]
    image_info = [
        {
            'img_tensor': image_tensor,
            'position': 0,  # Position in the text
        }
    ]
    
    # Prepare model input
    with torch.no_grad():
        # Prepare inputs
        text, vision_x = "", None
        
        # Handle direct tensor input
        text = prompt
        vision_tensors = [img_info['img_tensor'] for img_info in image_info]
        vision_x = torch.cat(vision_tensors, dim=1).unsqueeze(0)
        
        # Add image placeholder to text
        text = "<image>" + global_image_padding_tokens[0] + "</image>" + text
        
        # Tokenize text
        lang_x = global_tokenizer(
            text, max_length=2048, truncation=True, return_tensors="pt"
        )['input_ids'].to(device)
        
        # Move vision_x to device
        vision_x = vision_x.to(device)
        
        # Generate response
        generation = global_model.generate(lang_x, vision_x)
        response = global_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        
    return response

def create_multiplanar_view(volume_data, slice_idx=None):
    """Create multiplanar views (axial, coronal, sagittal) from volume data."""
    if slice_idx is None:
        # Default to middle slices
        slice_idx = {
            'axial': volume_data.shape[0] // 2,
            'coronal': volume_data.shape[1] // 2,
            'sagittal': volume_data.shape[2] // 2
        }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view (top-down)
    axes[0].imshow(volume_data[slice_idx['axial'], :, :], cmap='gray')
    axes[0].set_title(f'Axial (slice {slice_idx["axial"]})')
    axes[0].axis('off')
    
    # Coronal view (front-back)
    axes[1].imshow(volume_data[:, slice_idx['coronal'], :], cmap='gray')
    axes[1].set_title(f'Coronal (slice {slice_idx["coronal"]})')
    axes[1].axis('off')
    
    # Sagittal view (side)
    axes[2].imshow(volume_data[:, :, slice_idx['sagittal']], cmap='gray')
    axes[2].set_title(f'Sagittal (slice {slice_idx["sagittal"]})')
    axes[2].axis('off')
    
    # Save figure to bytes
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to PIL Image
    img = Image.open(buf)
    return img

# Main Gradio interface components
def process_medical_image(file_path, view_mode="2D/Single Slice"):
    """Process uploaded medical image file and return visualizations."""
    file_ext = os.path.splitext(file_path.name)[1].lower()
    
    # Determine file type and read image
    if file_ext in ['.dcm', '.dicom']:
        # DICOM file
        volume_data, metadata, is_3d, depth = read_dicom(file_path.name)
    elif file_ext in ['.nii', '.nifti', '.nii.gz']:
        # NIFTI file
        volume_data, metadata, is_3d, depth = read_nifti(file_path.name)
    else:
        return None, None, f"Unsupported file format: {file_ext}", [], False, 0, 0, 0
    
    if volume_data is None:
        return None, None, "Failed to read the medical image file.", [], False, 0, 0, 0
    
    # Prepare metadata display
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    
    # Determine visualization based on dimensionality and view mode
    if is_3d and view_mode in ["3D/Multiplanar", "3D/Volume"]:
        # Default to middle slices for multiplanar view
        middle_axial = volume_data.shape[0] // 2
        middle_coronal = volume_data.shape[1] // 2 if volume_data.shape[1] > 1 else 0
        middle_sagittal = volume_data.shape[2] // 2 if volume_data.shape[2] > 1 else 0
        
        # Create multiplanar view
        multiplanar_img = create_multiplanar_view(volume_data, {
            'axial': middle_axial,
            'coronal': middle_coronal,
            'sagittal': middle_sagittal
        })
        
        # Generate slice images for the slider
        axial_slices = [slice_to_pil(volume_data[i, :, :]) for i in range(volume_data.shape[0])]
        
        return multiplanar_img, volume_data, metadata_str, axial_slices, is_3d, depth, middle_axial, middle_coronal, middle_sagittal
    else:
        # 2D single slice view
        if is_3d:
            # For 3D, default to middle axial slice
            middle_slice = volume_data.shape[0] // 2
            slice_img = slice_to_pil(volume_data[middle_slice, :, :])
            
            # Generate all slices for the slider
            all_slices = [slice_to_pil(volume_data[i, :, :]) for i in range(volume_data.shape[0])]
            
            return slice_img, volume_data, metadata_str, all_slices, is_3d, depth, middle_slice, 0, 0
        else:
            # For 2D, just use the single slice
            slice_img = slice_to_pil(volume_data[0, :, :])
            return slice_img, volume_data, metadata_str, [slice_img], False, 1, 0, 0, 0

def update_view(volume_data, view_type, axial_pos, coronal_pos, sagittal_pos):
    """Update the view based on slice position changes."""
    if volume_data is None:
        return None
    
    # Convert numpy array if needed
    if isinstance(volume_data, list):
        return None
    
    try:
        # Ensure valid slice positions
        axial_pos = min(max(0, axial_pos), volume_data.shape[0]-1)
        
        max_coronal = volume_data.shape[1]-1 if volume_data.shape[1] > 1 else 0
        coronal_pos = min(max(0, coronal_pos), max_coronal)
        
        max_sagittal = volume_data.shape[2]-1 if volume_data.shape[2] > 1 else 0
        sagittal_pos = min(max(0, sagittal_pos), max_sagittal)
        
        if view_type == "Axial":
            return slice_to_pil(volume_data[axial_pos, :, :])
        elif view_type == "Coronal":
            return slice_to_pil(volume_data[:, coronal_pos, :])
        elif view_type == "Sagittal":
            return slice_to_pil(volume_data[:, :, sagittal_pos])
        else:  # Multiplanar
            return create_multiplanar_view(volume_data, {
                'axial': axial_pos,
                'coronal': coronal_pos,
                'sagittal': sagittal_pos
            })
    except Exception as e:
        print(f"Error updating view: {e}")
        return None

def process_chat(message, history, volume_data, view_type, axial_pos, coronal_pos, sagittal_pos):
    """Process chat message and integrate with RadFM inference."""
    if global_model is None:
        return "Model not loaded. Please load the model first."
    
    if volume_data is None:
        return "Please upload an image first."
    
    try:
        # Get current view for the model
        if view_type == "Axial":
            current_slice = volume_data[axial_pos, :, :]
        elif view_type == "Coronal":
            current_slice = volume_data[:, coronal_pos, :]
        elif view_type == "Sagittal":
            current_slice = volume_data[:, :, sagittal_pos]
        else:
            # For multiplanar, use axial slice for now
            current_slice = volume_data[axial_pos, :, :]
        
        # Run inference
        response = run_inference(current_slice, message)
        return response
    except Exception as e:
        print(f"Error processing chat: {e}")
        return f"Error processing request: {str(e)}"

# Define the Gradio application
def create_demo():
    with gr.Blocks(title="RadFM Medical Imaging Demo") as demo:
        gr.Markdown(
            """
            # RadFM Comprehensive Medical Imaging Demo
            
            This demo allows you to:
            1. Upload 2D or 3D medical images in DICOM or NIFTI format
            2. View slices in Axial, Coronal, and Sagittal orientations
            3. Navigate through volumes with sliders
            4. Chat with the RadFM model about your medical images
            
            **Note:** You need to load the model first. The model checkpoint should be placed in the Quick_demo folder.
            """
        )
        
        # State variables to track across interactions
        volume_data_state = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                load_model_btn = gr.Button("Load Model")
                file_input = gr.File(label="Upload DICOM or NIFTI file")
                view_mode = gr.Radio(
                    ["2D/Single Slice", "3D/Multiplanar", "3D/Volume"],
                    label="View Mode",
                    value="2D/Single Slice"
                )
                view_type = gr.Radio(
                    ["Axial", "Coronal", "Sagittal", "Multiplanar"],
                    label="View Type",
                    value="Axial"
                )
                
                with gr.Accordion("Volume Navigation", open=False):
                    axial_slider = gr.Slider(0, 100, 0, step=1, label="Axial Slice Position")
                    coronal_slider = gr.Slider(0, 100, 0, step=1, label="Coronal Slice Position")
                    sagittal_slider = gr.Slider(0, 100, 0, step=1, label="Sagittal Slice Position")
                
                metadata_text = gr.Textbox(label="Metadata", lines=10)
                
            with gr.Column(scale=2):
                image_output = gr.Image(label="Medical Image Visualization")
                
                with gr.Accordion("Chat with RadFM", open=True):
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(label="Enter your question about the image")
                    clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        load_model_btn.click(fn=load_model_and_tokenizer, outputs=metadata_text)
        
        file_input.change(
            fn=process_medical_image,
            inputs=[file_input, view_mode],
            outputs=[
                image_output, 
                volume_data_state, 
                metadata_text,
                chatbot,  # Reset chatbot
                axial_slider,  # Update slider max value
                axial_slider,  # Update slider value
                coronal_slider,
                sagittal_slider
            ]
        )
        
        # Function to update slider ranges based on volume dimensions
        def update_slider_range(volume_data, view_mode):
            if volume_data is None:
                return gr.Slider.update(maximum=0, value=0), gr.Slider.update(maximum=0, value=0), gr.Slider.update(maximum=0, value=0)
            
            # Get volume dimensions
            depth = volume_data.shape[0] - 1
            height = volume_data.shape[1] - 1 if volume_data.shape[1] > 1 else 0
            width = volume_data.shape[2] - 1 if volume_data.shape[2] > 1 else 0
            
            # Default positions (middle slices)
            axial_pos = depth // 2
            coronal_pos = height // 2
            sagittal_pos = width // 2
            
            return (
                gr.Slider.update(maximum=depth, value=axial_pos),
                gr.Slider.update(maximum=height, value=coronal_pos),
                gr.Slider.update(maximum=width, value=sagittal_pos)
            )
        
        view_mode.change(
            fn=update_slider_range,
            inputs=[volume_data_state, view_mode],
            outputs=[axial_slider, coronal_slider, sagittal_slider]
        )
        
        # Update view when sliders change
        def slider_change(volume_data, view_type, axial_pos, coronal_pos, sagittal_pos):
            return update_view(volume_data, view_type, int(axial_pos), int(coronal_pos), int(sagittal_pos))
        
        axial_slider.change(
            fn=slider_change,
            inputs=[volume_data_state, view_type, axial_slider, coronal_slider, sagittal_slider],
            outputs=image_output
        )
        
        coronal_slider.change(
            fn=slider_change,
            inputs=[volume_data_state, view_type, axial_slider, coronal_slider, sagittal_slider],
            outputs=image_output
        )
        
        sagittal_slider.change(
            fn=slider_change,
            inputs=[volume_data_state, view_type, axial_slider, coronal_slider, sagittal_slider],
            outputs=image_output
        )
        
        view_type.change(
            fn=slider_change,
            inputs=[volume_data_state, view_type, axial_slider, coronal_slider, sagittal_slider],
            outputs=image_output
        )
        
        # Chat functionality
        def user_message_and_response(message, history, volume_data, view_type, axial_pos, coronal_pos, sagittal_pos):
            history.append([message, None])
            response = process_chat(message, history, volume_data, view_type, axial_pos, coronal_pos, sagittal_pos)
            history[-1][1] = response
            return "", history
        
        msg.submit(
            fn=user_message_and_response,
            inputs=[msg, chatbot, volume_data_state, view_type, axial_slider, coronal_slider, sagittal_slider],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(lambda: [], outputs=chatbot)
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True) 