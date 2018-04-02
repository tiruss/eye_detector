import numpy as np
import cv2

from skimage.feature import hog

# Hog Function
def get_hog_featrues(img, orient, pix_per_cell, cell_per_block,  vis=False, feature_vec=True):

    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False, visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False, visualise=False,
                       feature_vector=feature_vec)
        return features


# Resize and Flatten image
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


# Compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    ch1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    ch3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

    return hist_features


# Extract features
def extract_features(imgs, spatial_size=(16, 16), hist_bins=32, hist_range=(0, 256),
                     orient=9, pixel_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    # image = mpimg.imread(img)
    for img in imgs:
        feature_image = np.copy(img)

        # Get color feature
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Get color histogram features
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Get HOG features
        if hog_channel == "ALL":
            hog_features = []
            for channel in range(feature_image.shape[2]):  # R G B
                hog_features.append(get_hog_featrues(feature_image[:, :, channel],
                                                     orient, pixel_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)  # Flatten
        else:
            hog_features = get_hog_featrues(feature_image[:, :, hog_channel],
                                            orient, pixel_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        # Concatenate features
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))

    return features


# Function to extract features from a single image
def extract_features_single(imgs, spatial_size=(16, 16), hist_bins=32, hist_range=(0, 256),
                            orient=9, pixel_per_cell=8, cell_per_block=2, hog_channel=0):
    features = []
    # image = mpimg.imread(img)
    feature_image = np.copy(imgs)

    # Get color feature
    spatial_features = bin_spatial(feature_image, size=spatial_size)

    # Get color histogram features
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

    # Get HOG features
    if hog_channel == "ALL":
        hog_features = []
        for channel in range(feature_image.shape[2]):  # R G B
            hog_features.append(get_hog_featrues(feature_image[:, :, channel],
                                                 orient, pixel_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)  # Flatten
    else:
        hog_features = get_hog_featrues(feature_image[:, :, hog_channel],
                                        orient, pixel_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
    # Concatenate Features
    features.append(np.concatenate((spatial_features, hist_features, hog_features)))

    return features


def sliding_window(img, x_start, y_start, xy_window=(16, 16), xy_overlap=(0.5, 0.5)):

    # Compute the span of the region to be searched
    xspan = x_start[1] - x_start[0]
    yspan = y_start[1] - y_start[0]

    # Compute the number of pixels per step
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    print("x pix per step : ", nx_pix_per_step)
    print("y pix per step : ", ny_pix_per_step)

    # Compute the number of windows
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / nx_pix_per_step) - 1
    print("number of X windows : ", nx_windows)
    print("number of Y windows : ", ny_windows)
    print("Total number of windows : ", nx_windows * ny_windows)

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            start_x = xs * nx_pix_per_step + x_start[0]
            end_x = start_x + xy_window[0]
            start_y = ys * ny_pix_per_step + y_start[0]
            end_y = start_y + xy_window[1]

            window_list.append(((start_x, start_y), (end_x, end_y)))

    return window_list


# Search and Classify
def search_windows(img, windows, clf, scaler, spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9, pixel_per_cell=8, cell_per_block=2,
                   hog_channel=0):
    # Positive detection windows
    on_windows = []

    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                              window[0][0]:window[1][0]], (64, 64))

        features = extract_features_single(test_img, spatial_size=spatial_size, hist_bins=hist_bins,
                                           hist_range=hist_range, orient=orient,
                                           pixel_per_cell=pixel_per_cell,
                                           cell_per_block=cell_per_block, hog_channel=hog_channel)

        # Scale extracted features
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict
        prediction = clf.predict(test_features)

        # Save window if positive (prediction == 1)
        if prediction == 1:
            on_windows.append(window)

    return on_windows

# Heatmap
def heatmap(heatmap_image, windows):

    for window in windows:
        # print(window[0][1],window[1][1], window[0][0],window[1][0])
        heatmap_image[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 10

    # plt.imshow(heatmap_img)

    return heatmap_image


# Merge windows that locate nearby others
def apply_threshold(heatmap, threshold):
    new_heatmap = np.copy(heatmap)

    # Zero out pixels below the threshold
    new_heatmap[new_heatmap <= threshold] = 0

    # Return threshold map
    return new_heatmap

def draw_labeled_bboxes(img, labels):
    for eye_img in range(1, labels[1] + 1):
        nonzero = (labels[0] == eye_img).nonzero()

        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        bbox = ((np.min(nonzero_x), np.min(nonzero_y)),
                (np.max(nonzero_x), np.max(nonzero_y)))

        cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 3)

    return img

