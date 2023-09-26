"""
This is the implementation for 16822 Assigment 1: Q1
Topic: Affine Rectification
"""
# Third Party
import os
import cv2
import argparse
import numpy as np
import random
from tabulate import tabulate

from utils import MyWarp

def write_img(output_path, image):
    try:
        # Attempt to save the image
        cv2.imwrite(output_path, image)
        print(f"Image saved successfully to {output_path}")
    except Exception as e:
        print(f"Error while saving the image: {str(e)}")

def cross_product_mat(vec):
    """
    Create a skew symmetric cross product matrix.

    Args:
        line_coeff (np.ndarray): [3,1] coefficients of line
    """
    return np.array([
    [0, -vec[2], vec[1]],
    [vec[2], 0, -vec[0]],
    [-vec[1], vec[0], 0]
    ])

def calc_affine_H(gt_points_homog_fit, overlay_img, file_name, colors):
    ideal_pts = []
    for i in range(0, gt_points_homog_fit.shape[0], 4):
        # Find the line coefficients and display them
        line_coeffs_1 = np.cross(gt_points_homog_fit[i+1], gt_points_homog_fit[i])
        line_coeffs_1 /= line_coeffs_1[-1]
        line_coeffs_2 = np.cross(gt_points_homog_fit[i+3], gt_points_homog_fit[i+2])
        line_coeffs_2 /= line_coeffs_2[-1]
        cv2.line(img_overlay_lines, gt_points[i].astype("int"), gt_points[i+1].astype("int"), color = colors[i//4], thickness=2)
        cv2.line(img_overlay_lines, gt_points[i+3].astype("int"), gt_points[i+2].astype("int"), color = colors[i//4], thickness=2)

        # # Assert that the points are on the line for validation
        # assert gt_points_homog[i] @ line_coeffs_1 == 0
        # assert gt_points_homog[i+2] @ line_coeffs_2 == 0

        # Find the imaged ideal points
        ideal_pt = cross_product_mat(line_coeffs_1) @ line_coeffs_2
        ideal_pt /= ideal_pt[-1]
        ideal_pts.append(ideal_pt)

    # Display/save the lines
    # cv2.imshow("Annotated Lines", img_overlay_lines)
    # cv2.waitKey()
    write_img(f"{file_name}_annotated_lines.jpg", overlay_img)

    # Image of the line at infinity
    imaged_l_inf = cross_product_mat(ideal_pts[0]) @ np.array(ideal_pts[1])
    imaged_l_inf /= imaged_l_inf[-1]

    # Calculate H from the imaged line at infinity
    H       = np.eye(3)
    H[2, :] = imaged_l_inf

    return H

def calc_angle_btwn_lines(l1, l2):
    return np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2))

def evaluate_angles(H, test_points, img, img_rect, file_stem, colors):
    """
    Evaluate the angles for the held out test points.
    Display a table of before and after H application.

    Args:
        H (np.ndarray): Homography matrix (3x3)
        test_points (4*N, 3): Held out test points for validation
    """
    eval_list = [[], []]
    test_lines_overlay = img.copy()
    warp_lines_overlay = img_rect.copy()
    for i in range(0, test_points.shape[0], 4):
        # Calculate the angle before H application
        line_coeffs_1 = np.cross(test_points[i+1], test_points[i])
        line_coeffs_1 /= line_coeffs_1[-1]
        line_coeffs_2 = np.cross(test_points[i+3], test_points[i+2])
        line_coeffs_2 /= line_coeffs_2[-1]
        angle_before  = calc_angle_btwn_lines(line_coeffs_1[:2], line_coeffs_2[:2])
        eval_list[0].append(angle_before)

        # Show test lines
        cv2.line(test_lines_overlay, test_points[i].astype("int")[:2],   test_points[i+1].astype("int")[:2], color = colors[i//4], thickness=2)
        cv2.line(test_lines_overlay, test_points[i+3].astype("int")[:2], test_points[i+2].astype("int")[:2], color = colors[i//4], thickness=2)

        # New lines
        l_prime_coeffs_1 = np.linalg.inv(H).T @ line_coeffs_1
        l_prime_coeffs_1 /= l_prime_coeffs_1[-1]
        l_prime_coeffs_2 = np.linalg.inv(H).T @ line_coeffs_2
        l_prime_coeffs_2 /= l_prime_coeffs_2[-1]
        angle_after = calc_angle_btwn_lines(l_prime_coeffs_1[:2], l_prime_coeffs_2[:2])
        eval_list[1].append(angle_after)

        # Show lines after rectification
        points_T  = H @ np.stack((test_points[i:i+5, :])).T
        points_T /= points_T[-1]
        cv2.line(warp_lines_overlay, points_T[:, 0].astype("int")[:2], points_T[:, 1].astype("int")[:2], color = colors[i//4], thickness=2)
        cv2.line(warp_lines_overlay, points_T[:, 2].astype("int")[:2], points_T[:, 3].astype("int")[:2], color = colors[i//4], thickness=2)

    # Display the table
    print(tabulate(eval_list, headers=["Before", "After"]))
    with open(f"{file_stem}_angles.txt", "w") as f:
        f.write(tabulate(eval_list, headers=["Before", "After"]))

    # Save the overlay images
    write_img(f"{file_stem}_test_lines.jpg", test_lines_overlay)
    write_img(f"{file_stem}_rect_test_lines.jpg", warp_lines_overlay)
    # cv2.imshow("sfsdf", warp_lines_overlay)
    # cv2.waitKey()

    
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A1:Q1 Driver", None, "")
    parser.add_argument("-i", "--img_file", default="data/q1/chess1.jpg")
    parser.add_argument("-a", "--annotation_file", default="data/annotation/q1_annotation.npy")
    parser.add_argument("-o", "--output_dir", default="output")
    args = parser.parse_args()

    # Load in the in the image of interest:
    img = cv2.imread(args.img_file)
    # Load in the annotations of interest:
    with open(args.annotation_file,'rb') as f:
      q1_annotation = np.load(f, allow_pickle=True)
    item_of_interest = args.img_file.split("/")[-1].split(".")[0]

    output_path      = f"{args.output_dir}/q1_{item_of_interest}"

    gt_points = np.array(q1_annotation.item().get(item_of_interest))
    gt_points_homog = np.hstack((gt_points, np.ones((gt_points.shape[0], 1))))

    # For gt points, each pair is a line and each pair of lines are parallel
    gt_points_homog_fit = gt_points_homog[0:8]
    gt_points_homog_test = gt_points_homog[8:]

    img_overlay_lines = img.copy()
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(gt_points.shape[0]//4)]

    # Get the homography and get the rectified image
    H_a = calc_affine_H(gt_points_homog_fit, img_overlay_lines, item_of_interest, colors)
    np.save(f"{output_path}_Ha", H_a)
    img_rect = MyWarp(img, H_a)
    # cv2.imshow("Rectified Image", img_rect)
    # cv2.waitKey()

    # Evalaute angles
    evaluate_angles(H_a, gt_points_homog_test, img, img_rect, output_path, colors)

    # Write the rectified image
    write_img(f"{output_path}_rectified.jpg", img_rect)

    # Psuedo Unit Test
    print(f"Transformed H maps imged line at infinity to line at infinity: {np.linalg.inv(H_a).T @ H_a[2, :]}")