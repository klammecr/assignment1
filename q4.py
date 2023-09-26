# Third Party
import argparse
import numpy as np
import cv2
import random

# In House
from q1 import evaluate_angles, write_img, overlay_img
from utils import MyWarp

def calc_H(pts, img, output_path, colors):
    # Here we need at least 5 pairs of perpendicular lines
    if pts.shape[0] < 20:
        raise ValueError("More points are required in order to run this method")
    
    N = pts.shape[0] // 4

    A = np.zeros((N,6))
    for i in range(0, pts.shape[0], 4):
        # Calculation
        l = np.cross(pts[i], pts[i + 1])
        l /= l[-1]
        m = np.cross(pts[i + 2], pts[i + 3])
        m /= m[-1]

        a_coeff = l[0] * m[0]
        b_coeff = (l[1]*m[0] + l[0]*m[1])/2
        c_coeff = l[1] * m[1]
        d_coeff = (l[2] * m[0] + l[0] * m[2])/2
        e_coeff = (l[2] * m[1] + l[1] * m[2]) / 2
        f_coeff = l[2] * m[2]
        A[i] = np.array([a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff])

    U, S, Vt = np.linalg.svd(A)
    x = Vt[np.argmin(S)]
    C = np.array([[x[0],  x[1]/2, x[3]/2],
                 [x[1]/2, x[2],   x[4]/2],
                 [x[3]/2, x[4]/2, x[5]]])
    
    U, D, Ut = np.linalg.svd(C)

    H = U @ D

    return H

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A1:Q4 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q1/tiles5.jpg")
    parser.add_argument("-a2", "--annotation_file", default="data/annotation/q2_annotation.npy")
    parser.add_argument("-o", "--output_dir", default="output/q2")
    args = parser.parse_args()

    # Load in the in the image of interest:
    img = cv2.imread(args.img_file)

    # Load in the annotations of interest:
    with open(args.annotation_file,'rb') as f:
      q2_annotation = np.load(f, allow_pickle=True)
    item_of_interest = args.img_file.split("/")[-1].split(".")[0]
    output_path      = f"{args.output_dir}/{item_of_interest}"

    gt_points = np.array(q2_annotation.item().get(item_of_interest))
    gt_points_homog = np.hstack((gt_points, np.ones((gt_points.shape[0], 1))))
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(gt_points.shape[0]//4)]

    # For gt points, each pair is a line and each pair of lines are parallel
    gt_points_homog_fit = gt_points_homog[0:8]
    gt_points_homog_test = gt_points_homog[8:]

    # Q2
    # Display annotations
    overlay_img(img, gt_points_homog_fit, output_path, "annotated", colors)

    # Affine Rectification
    H_a = np.load(f"{args.output_dir}/q1_{item_of_interest}_Ha.npy")
    aff_rect_img = MyWarp(img, H_a)

    # Calcualte test points after affine rectification
    test_pts = H_a @ gt_points_homog_test.T
    test_pts /= test_pts[-1]

    # Calculate similarity
    H_s = calc_H(test_pts.T, aff_rect_img.copy(), output_path, colors)
    sim_rect_img = MyWarp(img, H_s@ H_a)
    write_img(f"{output_path}_sim_rect.jpg", sim_rect_img)

    # Evaluate the angles after rectification
    evaluate_angles(H_s, test_pts.T, aff_rect_img, sim_rect_img, output_path, colors)

    # Display similarity transform
    # sim_rect_pts = H_s @ test_pts
    # sim_rect_pts = sim_rect_pts.T
    # for i in range(0, sim_rect_pts.shape[0], 4):
    #     cv2.line(sim_rect_img, sim_rect_pts[i+1][:2].astype("int"), sim_rect_pts[i][:2].astype("int"), colors[i//4])
    #     cv2.line(sim_rect_img, sim_rect_pts[i+3][:2].astype("int"), sim_rect_pts[i+2][:2].astype("int"), colors[i//4])
    # write_img(f"{output_path}_similarity_overlay.jpg", sim_rect_img)

   
    
