# Third Party
import argparse
import numpy as np
import cv2
import random

# In House
from q1 import evaluate_angles, write_img
from utils import MyWarp

def calc_similarity_H(pts, overlay_img, output_path, colors):
    if pts.shape[0] != 8:
        raise ValueError("Too many points")

    A = np.zeros((2, 3))
    for i in range(0, pts.shape[0], 4):
        # Calculation
        l = np.cross(pts[i], pts[i + 1])
        l /= l[-1]
        m = np.cross(pts[i + 2], pts[i + 3])
        m /= m[-1]
        A[i//4] = np.array([l[0] * m[0], l[0]*m[1] + l[1]*m[0], l[1]*m[1]])

        # Display
        cv2.line(overlay_img, pts[i+1][:2].astype("int"), pts[i][:2].astype("int"), colors[i//4])
        cv2.line(overlay_img, pts[i+3][:2].astype("int"), pts[i+2][:2].astype("int"), colors[i//4])

    write_img(f"{output_path}_affine_overlay.jpg", overlay_img)

    # Find S
    U, S, Vh = np.linalg.svd(A)
    s = Vh[np.argmin(S)]
    s_mat = np.array([
        [s[0], s[1]],
        [s[1], s[2]]
    ])
    s_mat /= s_mat[-1, -1]

    # Find H
    U, S, Vh = np.linalg.svd(s_mat)
    H_s = np.eye(3)
    H_s[:2, :2] = np.linalg.inv(np.diag(S**0.5)) @ Vh
    
    return H_s
   
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A1:Q1 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q1/facade.jpg")
    parser.add_argument("-a2", "--annotation_file", default="data/annotation/q2_annotation.npy")
    parser.add_argument("-o", "--output_dir", default="output")
    args = parser.parse_args()

    # Load in the in the image of interest:
    img = cv2.imread(args.img_file)

    # Load in the annotations of interest:
    with open(args.annotation_file,'rb') as f:
      q2_annotation = np.load(f, allow_pickle=True)
    item_of_interest = args.img_file.split("/")[-1].split(".")[0]
    output_path      = f"{args.output_dir}/q2_{item_of_interest}"

    gt_points = np.array(q2_annotation.item().get(item_of_interest))
    gt_points_homog = np.hstack((gt_points, np.ones((gt_points.shape[0], 1))))
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(gt_points.shape[0]//4)]

    # For gt points, each pair is a line and each pair of lines are parallel
    gt_points_homog_fit = gt_points_homog[0:8]
    gt_points_homog_test = gt_points_homog[8:]

    # Q2

    # Affine Rectification
    H_a = np.load(f"{args.output_dir}/q1_{item_of_interest}_Ha.npy")
    affine_rect_img = MyWarp(img, H_a)

    # Calcualte test points after affine rectification
    test_pts = H_a @ gt_points_homog_test.T
    test_pts /= test_pts[-1]

    # Calculate similarity
    H_s = calc_similarity_H(test_pts.T, affine_rect_img.copy(), output_path, colors)
    sim_rect_img = MyWarp(img, H_s@ H_a)
    write_img(f"{output_path}_sim_rect.jpg", sim_rect_img)
    
    evaluate_angles(H_s, gt_points_homog_test, affine_rect_img, sim_rect_img, output_path, colors)
