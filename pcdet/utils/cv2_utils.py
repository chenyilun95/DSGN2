import numpy as np
import cv2

def draw_3d_boxes(img, boxes_corners, color=(36,255,12)):
    img = img.copy()
    boxes_corners = boxes_corners.round().astype(int)
    box_lines = np.array([[2,3],[0,3],[4,5],[4,7],[5,6],[6,7],[0,4],[1,5],[2,6],[3,7]])
    for i in range(len(boxes_corners)):
        for s, t in box_lines:
            cv2.line(img, tuple(boxes_corners[i, s].tolist()), tuple(boxes_corners[i, t].tolist()), color=color)
    return img

def draw_2d_boxes(img, boxes_corners, color=(36,255,12)):
    img = img.copy()
    color = tuple(color)
    if boxes_corners.shape[1] == 8: # 8 point corners
        boxes_corners = boxes_corners.round().astype(int)
        boxes_corners = np.concatenate([boxes_corners.min(axis=1), boxes_corners.max(axis=1)], axis=1)
    for i in range(len(boxes_corners)):
        x0, y0, x1, y1 = boxes_corners[i]
        cv2.rectangle(img, (round(x0), round(y0)), (round(x1), round(y1)), color, 2)
    return img
