# Third Party
import numpy as np
import argparse
import cv2
import random

# In House
from q1 import cross_product_mat, write_img


def overlay_correspondences(img, pts, colors, out_fp):
    for i in range(pts.shape[0]):
        cv2.circle(img, tuple(pts[i].astype("int")), radius = 10, color=colors[i], thickness=-1)
    # cv2.imshow("Overlayed Points", img)
    # cv2.waitKey()
    write_img(out_fp, img)

def calc_H_correspond(src_pt, tgt_pt):
    # Create constraints in A matrix
    A = np.zeros((src_pt.shape[0]*2, 9))
    for i in range(src_pt.shape[0]):
        x_coeff = np.zeros((3, 9))
        x_coeff[0, :3]  = src_pt[i]
        x_coeff[1, 3:6] = src_pt[i]
        x_coeff[2, 6:]  = src_pt[i]
        x_dash_cross = cross_product_mat(tgt_pt[i])

        # Each correspondence yields 2 constraints, 3rd is a linear combination
        # of the first two
        A_idx = 2*i
        constraints = x_dash_cross.T @ x_coeff
        A[A_idx:A_idx+2] = constraints[:2]
    
    # SVD then unflatten to get H
    U, S, Vt = np.linalg.svd(A)
    # x = U[:, np.argmin(S)]
    # H = np.array([
    #     [x[0], x[1], x[2]],
    #     [x[3], x[4], x[5]],
    #     [x[6], x[7], 1.]
    # ])
    x = Vt[-1]
    H = x.reshape((3,3))
    return H

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser("GeoViz A1:Q3 Driver", None, "")
    parser.add_argument("-s", "--src_img_file", default="data/q3/desk-normal.png")
    parser.add_argument("-t", "--target_img_file", default="data/q3/desk-perspective.png")
    parser.add_argument("-ap", "--annotation_file_perspective", default="data/annotation/q3_annotation.npy")
    parser.add_argument("-o", "--output_dir", default="output/q3")
    args = parser.parse_args()

    # Load in the in the image of interest and annotations:
    src_img = cv2.imread(args.src_img_file)
    tgt_img = cv2.imread(args.target_img_file)

    # Create or load annotations
    H, W, _ = src_img.shape
    src_annotation_pts = np.array([
        [0, 0],
        [W-1, 0],
        [W-1, H-1],
        [0, H-1]
    ])

    # TODO: Load possible different source annotations?

    with open(args.annotation_file_perspective,'rb') as f:
        annotations = np.load(f, allow_pickle=True)
    item_of_interest = args.target_img_file.split("/")[-1].split(".")[0]
    output_path      = f"{args.output_dir}/{item_of_interest}"
    tgt_annotation_pts = annotations.item()[item_of_interest]
    
    # Optional: Visualize the correspondences
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(4)]
    write_img(f"{output_path}_perspective.jpg", tgt_img)
    overlay_correspondences(src_img.copy(), src_annotation_pts, colors, f"{output_path}_src_annotation.jpg")
    overlay_correspondences(tgt_img.copy(), tgt_annotation_pts, colors, f"{output_path}_tgt_annotation.jpg")

    # Find H then warp
    tgt_annotation_pts_homog = np.hstack((tgt_annotation_pts, np.ones((len(tgt_annotation_pts), 1))))
    src_annotation_pts_homog = np.hstack((src_annotation_pts, np.ones((len(src_annotation_pts), 1))))
    H = calc_H_correspond(src_annotation_pts_homog, tgt_annotation_pts_homog)

    # Blend images
    overlay_img = tgt_img.copy()
    msk = np.ones_like(src_img)
    res_img = cv2.warpPerspective(src_img, H, (tgt_img.shape[1], tgt_img.shape[0]))
    warp_msk = cv2.warpPerspective(msk, H, (tgt_img.shape[1], tgt_img.shape[0]))
    msk = (np.all(warp_msk == 1, 2))
    overlay_img[warp_msk == 1] = res_img[warp_msk == 1]

    write_img(f"{output_path}_warp_overlay.jpg", overlay_img)