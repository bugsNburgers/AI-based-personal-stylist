import numpy as np
import cv2


def polygon_to_mask(segmentation, height, width):
    """
    Converts DeepFashion2 polygon segmentation to a binary mask.

    segmentation: list of [x1,y1,x2,y2,...]
    height, width: image dimensions
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if segmentation is None or len(segmentation) == 0:
        return mask

    # DeepFashion2 stores segmentation as list of polygons
    pts = np.array(segmentation).reshape(-1, 2).astype(np.int32)

    cv2.fillPoly(mask, [pts], 255)

    return mask

def extract_garment(image, mask, bbox):
    """
    Extract garment using mask + bounding box.
    Returns RGBA image with transparent background.
    """
    x1, y1, x2, y2 = bbox

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Create alpha channel
    alpha = np.where(crop_mask > 0, 255, 0).astype(np.uint8)

    # Convert to RGBA
    rgba = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha

    return rgba
