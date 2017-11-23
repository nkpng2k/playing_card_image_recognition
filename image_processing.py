import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, splitext
from collections import Counter
from skimage import color, filters, io, transform, feature, exposure
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CardImageProcessing(object):
    """
    Class that will process card images within a file
    returns the processed images
    NOTE: must run file_info(self, file_path) method
          before any other preprocessing

    INPUT: directory with images
    ATTRIBUTES: self.raw_img - raw images read into list
                self.files - list of files
                self.file_names - list of files sans extensions
                self.file_ext - file extension used to parse files
    METHODS: label_images (returns: list of labels)
             vectorize_images (returns: array of 1-D vectors,
                               raw images vectorized into 1-D array)
    """

    def __init__(self):
        self.file_path = None
        self.files = None
        self.file_names = None
        self.file_ext = None
        self.corner_pca = None
        self.card_pca = None
        self.corner_scaler = None
        self.card_scaler = None
        self.num_corner_comp = None
        self.num_card_comp = None

    def _read_in_images(self):
        raw_list = []
        grey_list = []
        for f in self.files:
            img = io.imread(self.file_path+'/'+f)
            raw_list.append(img)
        for img in raw_list:
            grey = color.rgb2grey(img)
            grey_list.append(grey)
        return raw_list, grey_list

    def _calculate_intersections(self, cropped_img):
        """
        Identifies probabilistic hough lines and
        uses those to calculate transform coordinates
        INPUT: Single image cropped to corners of playing card
        OUTPUT: Single list of coordinates, unordered (x, y)
        """
        edges = feature.canny(cropped_img,
                              low_threshold=0.2, high_threshold=1)
        lines = transform.probabilistic_hough_line(edges,
                                                   threshold=50,
                                                   line_length=275,
                                                   line_gap=10)

        set_slopes, set_lines = set(), set()
        pos_slope, neg_slope = [], []
        for line in lines:
            p0, p1 = line
            slope, intercept, _, _, _ = stats.linregress([p0[0], p1[0]],
                                                         [p0[1], p1[1]])
            if True not in np.isclose(round(slope, 2),
                                      list(set_slopes), atol=1e-02):
                set_slopes.add(round(slope, 2))
                set_lines.add(line)
                if slope > 0:
                    pos_slope.append((round(slope, 2), intercept))
                else:
                    neg_slope.append((round(slope, 2), intercept))

        coord_int = []
        for slope in pos_slope:
            coord1 = np.linalg.solve(np.array([[-slope[0], 1],
                                              [-neg_slope[0][0], 1]]),
                                     np.array([slope[1],
                                              neg_slope[0][1]]))
            coord2 = np.linalg.solve(np.array([[-slope[0], 1],
                                              [-neg_slope[1][0], 1]]),
                                     np.array([slope[1],
                                               neg_slope[1][1]]))
            coord_int.append(coord1)
            coord_int.append(coord2)

        if len(coord_int) < 4:
            coord_int = [[0, 0], [0, 93], [68, 0], [68, 93]]

        return np.array(coord_int)

    def _orient_intersection_coords(self, cropped, coord_int):
        """
        Identifies orientation of playing card.
        Designates coordinates from coord_int as
        top left (tr), top right (tr), bottom left (bl), bottom right (br)
        INPUT: Single image cropped to corners of playing card
               coord_int --> coordinates of intersection of
                             probabilistic hough lines
        OUTPUT: dst --> array ordered coordinates (tl, bl, br, tr),
                        specific order needed for skimage ProjectiveTransform
        """
        mask = coord_int == np.array([[0, 0], [0, 93], [68, 0], [68, 93]])
        if np.all(mask):
            tl, tr, bl, br = [0, 0], [68, 0], [0, 93], [68, 93]
            dst = np.array([tl, bl, br, tr])
            return dst

        xmin = coord_int[np.argmin(coord_int[:, 0]), :]
        xmax = coord_int[np.argmax(coord_int[:, 0]), :]
        ymin = coord_int[np.argmin(coord_int[:, 1]), :]
        ymax = coord_int[np.argmax(coord_int[:, 1]), :]

        if cropped.shape[0] < cropped.shape[1]:
            if xmin[1] > xmax[1]:
                tl, tr, bl, br = xmin, ymin, ymax, xmax
            else:
                tl, tr, bl, br = ymax, xmin, xmax, ymin
        else:
            if xmin[1] > xmax[1]:
                tl, tr, bl, br = ymin, xmax, xmin, ymax
            else:
                tl, tr, bl, br = xmin, ymin, ymax, xmax

        dst = np.array([tl, bl, br, tr])
        return dst

    # ------- NOTE: all public methods below this line --------

    def file_info(self, file_path):
        """
        Reads all images in a file.
        Identifies most common file extension as file to take as input
        INPUT: String --> filepath to directory
                          ('User/username/data/all_images').
                          Do not include "/" at end of filepath
        OUTPUT: list of raw images, converted to grey scale
        """
        onlyfiles = [f for f in listdir(file_path)
                     if isfile(join(file_path, f))]

        file_ext_count = Counter()
        for f in onlyfiles:
            fname, file_type = splitext(f)
            file_ext_count[file_type] += 1

        self.file_path = file_path
        self.file_ext = file_ext_count.most_common()[0][0]
        self.files = [f for f in onlyfiles if splitext(f)[1] == self.file_ext]
        self.file_names = [splitext(f)[0] for f in onlyfiles
                           if splitext(f)[1] == self.file_ext]

        raw_imgs = self._read_in_images()
        return raw_imgs

    def generate_labels(self, delimiter=None, labels=None):
        """
        will manually assign labels for each of the images or if no manual
        labels are provided will pull the characters up until
        a specified delimiter as the label

        INPUT: labels --> (list or tuples) optional, assign labels for images
                          tuple will have this order: (card type, card suit)
               delimiter --> (string) delimiter that is expected to separate
                             the card type and card suit.
                             Example: queen_heart.png - delimiter = '_'
        OUTPUT: 2 lists --> card type and card suit
        """
        card_type = []
        card_suit = []
        if labels is None:
            for name in self.file_names:
                card_type.append(name.split(delimiter)[0])
                card_suit.append(name.split(delimiter)[1])
        else:
            for tup in labels:
                card_type.append(tup[0])
                card_suit.append(tup[1])
        return card_type, card_suit

    def bounding_box_crop(self, images):
        """
        Detect edges, mask everything outside of edges to 0,
        determine coordinates for corners of card,
        crop box tangent to corners of card
        INPUT: List of raw images, grey scaled
        OUTPUT: List of cropped images. For playing cards,
        will crop to corners of card
        """
        cropped_list = []
        for img in images:
            edges = filters.thresholding.threshold_minimum(img)
            img[img < edges] = 0

            coords = np.argwhere(img > 0.9)

            miny, minx = coords.min(axis=0)
            maxy, maxx = coords.max(axis=0)

            cropped = img[miny:maxy, minx:maxx]

            cropped_list.append(cropped)

        return cropped_list

    def rotate_images(self, images):
        """
        Perform projective transform on grey scaled images
        INPUT: List of images. Must be cropped to bounding bounding box
        OUTPUT: List of images (2-D arrays), warped to vertical orientation.
        """
        warped_images, top_left_corner = [], []
        for img in images:
            intersect_coords = self._calculate_intersections(img)
            dst = self._orient_intersection_coords(img, intersect_coords)
            src = np.array([[0, 0], [0, 93], [68, 93], [68, 0]])
            persp_transform = transform.ProjectiveTransform()
            persp_transform.estimate(src, dst)
            warped = transform.warp(img, persp_transform,
                                    output_shape=(93, 68))
            warped_images.append(warped)
            top_left_corner.append(warped[:30, :15])

        return warped_images, top_left_corner

    def vectorize_images(self, images):
        """
        Generate HOG vectors for grey scaled images.
        INPUT: List of images. Images are array type.
        OUTPUT: vectorized_images --> list of 1-D arrays.
                                      Feature Vectors for each image
                hog_images --> list of 2-D arrays, HOG representation of images
        """
        vectorized_images, hog_images = [], []
        for img in images:
            vector, hog_image = feature.hog(img, orientations=10,
                                            pixels_per_cell=(3, 3),
                                            cells_per_block=(3, 3),
                                            block_norm='L2-Hys',
                                            visualise=True)
            vectorized_images.append(vector)
            hog_images.append(hog_image)
        return vectorized_images, hog_images

    def train_pca(self, corner_vectors, card_vectors):
        self.corner_pca = PCA(n_components=10)
        self.card_pca = PCA(n_components=10)

        self.corner_scaler = StandardScaler().fit(corner_vectors)
        self.card_scaler = StandardScaler().fit(card_vectors)

        self.corner_pca.fit(corner_vectors)
        self.card_pca.fit(card_vectors)

        for i in xrange(self.corner_pca.n_components_):
            if sum(self.corner_pca.explained_variance_ratio_[:i]) > 0.9:
                self.num_corner_comp = i
        for i in xrange(self.card_pca.n_components_):
            if sum(self.card_pca.explained_variance_ratio_[:i]) > 0.9:
                self.num_card_comp = i

    def reduce_dimensions(self, corner_vectors, card_vectors):
        trans_corner = self.card_scaler.transform(corner_vectors)
        trans_card = self.corner_scaler.transform(card_vectors)

        tl_pca = self.corner_pca.transform(trans_corner)
        card_pca = self.card_pca.transform(trans_card)

        return tl_pca[: self.num_corner_comp], card_pca[: self.num_card_comp]

    def training_images_pipe(self, filepath):
        """
        Single method for piping Images
        INTPUT: filepath to Images
        OUTPUT: lists of vectorized cards and labels
        """
        raw_imgs, grey_imgs = self.file_info(filepath)
        c_type, c_suit = self.generate_labels(delimiter='_')
        cropped_imgs = self.bounding_box_crop(grey_imgs)
        warped_imgs, tl_corner = self.rotate_images(cropped_imgs)
        vectorized_cards, hog_cards = self.vectorize_images(warped_imgs)
        vectorized_corner, hog_corner = self.vectorize_images(tl_corner)
        self.train_pca(vectorized_corner, vectorized_cards)
        corner_pca, card_pca = self.reduce_dimensions(vectorized_corner,
                                                      vectorized_cards)

        return card_pca, corner_pca, c_type, c_suit


if __name__ == "__main__":
    filepath = '/Users/npng/galvanize/playing_card_image_recognition/samples'
    card_process = CardImageProcessing()
    raw_imgs, grey_imgs = card_process.file_info(filepath)
    c_type, c_suit = card_process.generate_labels(delimiter='_')
    cropped_imgs = card_process.bounding_box_crop(grey_imgs)
    warped_imgs, tl_corner = card_process.rotate_images(cropped_imgs)
    vectorized_imgs, hog_imgs = card_process.vectorize_images(warped_imgs)
    vectorized_corner, hog_corner = card_process.vectorize_images(tl_corner)
    io.imshow(tl_corner[1])
    io.show()
    io.imshow(warped_imgs[1])
    io.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                                   sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(tl_corner[0], cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_corner[0],
                                                    in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

    fp2 = '/Users/npng/galvanize/playing_card_image_recognition/card_images'
    card_process = CardImageProcessing()
    raw_imgs, grey_imgs = card_process.file_info(fp2)
    results = card_process.training_images_pipe(fp2)
    c_type, c_suit = card_process.generate_labels(delimiter='_')
    cropped_imgs = card_process.bounding_box_crop(grey_imgs)
    warped_imgs, tl_corner = card_process.rotate_images(cropped_imgs)
    vectorized_imgs, hog_imgs = card_process.vectorize_images(warped_imgs)
    vectorized_corner, hog_corner = card_process.vectorize_images(tl_corner)
    io.imshow(tl_corner[20])
    io.show()
    io.imshow(warped_imgs[20])
    io.show()

"""
bottom of page
"""
