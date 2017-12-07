"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import data, filters, io, measure, transform, feature, exposure
from skimage.feature import corner_fast, corner_peaks, corner_harris
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import math
from scipy import stats

image = data.coins()
edges = filters.sobel(image)
io.imshow(edges)
io.show()

"""
train a model to classify cards
pass the model new image frames with a new card
have the model identify it and count
"""
%cd /Users/npng/galvanize/Dream_3_repository

#read in sample with grey scale -- Numpy values 0 = black, 1 = white
card_image = io.imread('samples/IMG_1197.JPG', as_grey = True)
card_image_2 = io.imread('samples/IMG_1198.JPG', as_grey = True)
card_photo = io.imread('card_photos/j_h.JPG', as_grey=True)

# filters.thresholding.threshold_minimum finds minimum value to separate edges
edges = filters.threshold_minimum(card_photo)
filtered = card_photo.copy()
filtered[filtered < edges] = 0
edges_2 = filters.threshold_minimum(card_image_2)
filtered_2 = card_image_2.copy()
filtered_2[filtered_2 < edges] = 0

io.imshow(filtered)
io.show()

coords = np.argwhere(filtered > 0.9)
miny, minx = coords.min(axis = 0)
maxy, maxx = coords.max(axis = 0)

cropped = filtered[miny:maxy,minx:maxx]

coords = np.argwhere(filtered_2 > 0.9)
miny, minx = coords.min(axis = 0)
maxy, maxx = coords.max(axis = 0)

cropped_2 = filtered_2[miny:maxy,minx:maxx]


edges = feature.canny(card_photo)
lines = transform.probabilistic_hough_line(edges, threshold=50, line_length=500, line_gap=10)
len(lines)
#NOTE: use np.polyfit(), np.roots() of two polyfits will return the intersections

set_slopes, set_lines = set(), set()
pos_slope, neg_slope = [], []
for line in lines:
    p0, p1 = line
    slope, intercept, _, _, _ = stats.linregress([p0[0], p1[0]], [p0[1], p1[1]])
    if True not in np.isclose(round(slope, 2), list(set_slopes), atol = 1e-02):
        set_slopes.add(round(slope, 2))
        set_lines.add(line)
        if slope > 0:
            pos_slope.append((round(slope, 2), intercept))
        else:
            neg_slope.append((round(slope, 2), intercept))

coord_int = []
for slope in pos_slope:
    coord1 = np.linalg.solve(np.array([[-slope[0], 1], [-neg_slope[0][0], 1]]), np.array([slope[1], neg_slope[0][1]]))
    coord2 = np.linalg.solve(np.array([[-slope[0], 1], [-neg_slope[1][0], 1]]), np.array([slope[1], neg_slope[1][1]]))
    coord_int.append(coord1)
    coord_int.append(coord2)

coord_int = np.array(coord_int)

xmin = coord_int[np.argmin(coord_int[:, 0]), :]
xmax = coord_int[np.argmax(coord_int[:, 0]), :]
ymin = coord_int[np.argmin(coord_int[:, 1]), :]
ymax = coord_int[np.argmax(coord_int[:, 1]), :]

if cropped_2.shape[0] < cropped_2.shape[1]:
    if coord_int[np.argmin(coord_int[:, 0]), :][1] > coord_int[np.argmax(coord_int[:, 0]), :][1]:
        tl, tr, bl, br = xmin, ymin, ymax, xmax
    else:
        tl, tr, bl, br = ymax, xmin, xmax, ymin
else:
    if coord_int[np.argmin(coord_int[:, 0]), :][1] > coord_int[np.argmax(coord_int[:, 0]), :][1]:
        tl, tr, bl, br = ymin, xmax, xmin, ymax
    else:
        tl, tr, bl, br = xmin, ymin, ymax, xmax


dst = np.array([tl, bl, br, tr])

xs = [x[0] for x in coord_int]
ys = [x[1] for x in coord_int]

src = np.array([[0, 0], [0, 93], [68, 93], [68, 0]])
test_coords = np.array([[-20, 421], [462, 1503], [1306, 1106], [904, -68]])

persp_transform = transform.ProjectiveTransform()
persp_transform.estimate(src, dst)
warped = transform.warp(cropped_2, persp_transform, output_shape = (93, 68))
io.imshow(warped)
plt.show()

fd, hog_image = feature.hog(warped, orientations = 10, pixels_per_cell = (3,3), cells_per_block = (2,2),
                            block_norm = 'L2-Hys', visualise = True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(warped, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()

fig, ax = plt.subplots()
ax.imshow(card_photo, cmap = plt.cm.gray)
for line in lines:
    p0, p1 = line
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
ax.scatter(xs, ys)
plt.show()










"""
Double card stuff
"""

dbl_card = io.imread('samples/IMG_1199.jpg', as_grey = True)

edges = filters.thresholding.threshold_otsu(dbl_card)
dbl_filtered = dbl_card.copy()
dbl_filtered[dbl_filtered < edges] = 0

io.imshow(dbl_filtered)
io.show()

#NOTE: need to use contour lines to generate bounding box

contours = measure.find_contours(cropped, .8, 'high')

contours
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(cropped, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
"""
Bottom of Page
"""
