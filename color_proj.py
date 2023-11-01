import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import sparse
from typing import Tuple
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csc_matrix
import argparse

# Converts the given coordinates to an index based on the width of the image
def idx_from_coord(x_coord, y_coord, y_dim):
    return x_coord * y_dim + y_coord

# Finds the adjacent coordinates to the given point within the specified parameters 
def adjacent_coords(x_coord, y_coord, x_dim, y_dim):
    adj = []
    for i in range(max(0, x_coord - 1), min(x_dim, x_coord +2)):
        for j in range(max(0, y_coord - 1), min(y_dim, y_coord + 2)):
            if (i != x_coord) or (j != y_coord):
                adj.append([i, j])
    return adj

# Transforms the input images from RGB to YUV and normalizes values to color images later
def transform_images(gray_img, scribbles_img):
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2YUV) / 255.0
    scribbles_img = cv2.cvtColor(scribbles_img, cv2.COLOR_RGB2YUV) / 255.0
    return gray_img, scribbles_img

# get the weights for each element in colArr based on colVal and sigmaVal
def calculate_weights(colVal, colArr, sigmaVal):
    wgts = np.zeros(len(colArr))

    if sigmaVal > 1e-3:
        diff = colArr - colVal
        squared_diff = diff ** 2
        exponent = -1 * squared_diff / (2 * sigmaVal * sigmaVal)
        wgts = np.exp(exponent)
    else:
        wgts.fill(1.)

    total_wgts = np.sum(wgts)
    norm_wgts = wgts / total_wgts

    return norm_wgts

# Populates the W_matrix, curr_vec, and next_vec based on gray_img and scribbles_img
def populate(x_dim, y_dim, W_matrix, curr_vec, next_vec, gray_img, scribbles_img):
    for i in tqdm(range(x_dim)):
        for j in range(y_dim):
            idx = idx_from_coord(i, j, y_dim)
            colVal = gray_img[i, j, 0]

            if scribbles_img[i, j, 0] > 1 - 1e-3 or (np.abs(gray_img[i, j] - scribbles_img[i, j]) > 1e-2).any():
                W_matrix[idx, idx] = 1
                curr_vec[idx] = scribbles_img[i, j, 1]
                next_vec[idx] = scribbles_img[i, j, 2]
            else:
                adj = adjacent_coords(i, j, x_dim, y_dim)
                colArr = np.array([gray_img[pos[0], pos[1], 0] for pos in adj])
                idxs = np.array([idx_from_coord(pos[0], pos[1], y_dim) for pos in adj])

                sigmaVal = np.std(colArr)
                wgts = calculate_weights(colVal, colArr, sigmaVal)

                W_matrix[idx, idxs] += -1 * wgts
                W_matrix[idx, idx] += 1

# Colorizes the image using the algoritm from the paper
def color_image(gray_img, scribbles_img) -> np.ndarray:
    gray_img, scribbles_img = transform_images(gray_img, scribbles_img)
    x_dim, y_dim = scribbles_img.shape[0], scribbles_img.shape[1]
    total_size = x_dim * y_dim
    W_matrix = lil_matrix((total_size, total_size), dtype=float)
    curr_vec = np.zeros(shape=(total_size))
    next_vec = np.zeros(shape=(total_size))
    populate(x_dim, y_dim, W_matrix, curr_vec, next_vec, gray_img, scribbles_img)
    final_result = np.zeros(shape=(x_dim, y_dim, 3))
    final_result[:, :, 0] = gray_img[:, :, 0]
    W_matrix = csc_matrix(W_matrix)
    u_vec = spsolve(W_matrix, curr_vec)
    v_vec = spsolve(W_matrix, next_vec)

    for idx, value in np.ndenumerate(final_result[:, :, 1]):
        i, j = idx
        index = idx_from_coord(i, j, y_dim)
        final_result[i, j, 1], final_result[i, j, 2] = u_vec[index], v_vec[index]

    final_result = (np.clip(final_result, 0., 1.) * 255).astype(np.uint8)
    final_result = cv2.cvtColor(final_result, cv2.COLOR_YUV2RGB)
    return final_result

def main(gray_img_path: str, scribbles_img_path: str) -> None:
    original_gray_img = cv2.cvtColor(cv2.imread(gray_img_path), cv2.COLOR_BGR2RGB)
    original_scribbles_img = cv2.cvtColor(cv2.imread(scribbles_img_path), cv2.COLOR_BGR2RGB)
    final_output = color_image(original_gray_img, original_scribbles_img)

    # Save the final result to a PNG file
    output_file_name = "colorized_output.png"
    Image.fromarray(final_output).save(output_file_name)
    print(f"Saved the final result to {output_file_name}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("gray_img_path")
    arg_parser.add_argument("scribbles_img_path")
    input_args = arg_parser.parse_args()
    main(input_args.gray_img_path, input_args.scribbles_img_path)