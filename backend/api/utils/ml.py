from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, UploadFile
from sqlalchemy.future import select
from PIL import Image
import numpy as np
import cv2
import patchify
import tensorflow as tf
import keras
from keras.api.utils import register_keras_serializable
from fastapi.responses import StreamingResponse
from io import BytesIO
from ...db.models.user import User


@register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.api.backend.flatten(y_true)
    y_pred_f = keras.api.backend.flatten(y_pred)
    intersection = keras.api.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.api.backend.sum(y_true_f) + keras.api.backend.sum(y_pred_f) + smooth)


# Load the model (ensure this matches your actual model path)
model = keras.api.models.load_model(
    'backend/machine_learning/tumor_segmentation_model.keras',
    custom_objects={'dice_loss': dice_loss}
)

smooth = 1e-15

cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16,
    "num_patches": (256**2) // (16**2),
    "flat_patches_shape": (
        (256**2) // (16**2),
        16*16*3
    )
}

async def tumor_detection(db: AsyncSession, user_id: int, file: UploadFile):
    # Get user info
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Read and preprocess the image
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.convert('RGB')
        img = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    original_size = img.shape[:2]  # Original image size
    image = cv2.resize(img, (cf["image_size"], cf["image_size"]))
    x = image / 255.0

    # Create patches and prepare for prediction
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    # Prediction
    try:
        pred = model.predict(patches, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    
    pred = np.reshape(pred, (cf["image_size"], cf["image_size"]))

    # Resize the prediction to match the original image size
    pred_resized = cv2.resize(pred, original_size[::-1])

    # Threshold the prediction to create a binary mask
    binary_mask = (pred_resized > 0.5).astype(np.uint8)  # Adjust threshold if needed

    # Create a red mask for the segmented area
    red_mask = np.zeros_like(img)
    red_mask[:, :, 0] = binary_mask * 255  # Red channel

    # Overlay the red mask on the original image
    overlayed_image = cv2.addWeighted(img, 1, red_mask, 0.5, 0)

    # Convert the overlayed image to bytes
    img_byte_arr = BytesIO()
    overlayed_image_pil = Image.fromarray(overlayed_image)
    overlayed_image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Return the result as a streaming response
    return StreamingResponse(img_byte_arr, media_type="image/png", headers={"Content-Disposition": "inline; filename=result.png"})
