import cv2
import numpy  as np

def four_point_transform(image, pts, target_size=None):
    """
    Perspective Transform
    Input:
        - image: raw  (numpy array)
        - pts: list 4 points from YOLO pose
    Output:
        - warped
    """
    (tl, tr, br, bl) = pts

    if target_size is None:
        # 1. Calculate the Width of the new image
        # Max (BR-BL) or (TR-TL)
        widthA = np.sqrt(((br[0]-bl[0])**2 + (br[1]-bl[1])**2))
        widthB = np.sqrt(((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))

        # 2. Calculate the Height of the new image
        # Max  (TR-BR) or (TL-BL)
        heightA  =  np.sqrt(((tr[0]-br[0])**2 +(tr[1]-br[1])**2))
        heightB  =  np.sqrt(((tl[0]-bl[0])**2 +(tl[1]-bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

    else:
        maxWidth, maxHeight = target_size
        
    # 3. Construct the Destination Points
    dst = np.array([
        [0,0],                    # Top-left corner
        [maxWidth-1, 0],          # Top-right corner
        [maxWidth-1,maxHeight-1], # Bottom-right corner
        [0, maxHeight-1]          # Bottom-left corner
    ], dtype="float32")

    src = np.array([tl, tr, br, bl], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

    return warped