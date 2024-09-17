from itertools import product
import numpy as np
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
import cv2

def binary_array_to_dec (bin_array):
    string = ''
    for byte in bin_array:
        for bit in byte:
            string += str (int (bit))
    dec = int (string, 2)
    return dec

def _order_polygon_points(x, y):
    """
    Orders polygon points into a counterclockwise order.
    x_p, y_p are the x and y coordinates of the polygon points.
    """
    angle = np.arctan2(y - np.average(y), x - np.average(x))
    ind = np.argsort(angle)
    return (x[ind], y[ind])


def four_point_transform(image, poly):
    """
    A perspective transform for a quadrilateral polygon.
    Slightly modified version of the same function from
    https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
    """
    pts = np.zeros((4, 2))
    pts[:, 0] = np.asarray(poly.exterior.coords)[:-1, 0]
    pts[:, 1] = np.asarray(poly.exterior.coords)[:-1, 1]
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = _order_polygon_points(pts[:, 0], pts[:, 1])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    # width_a = np.sqrt(((b_r[0] - b_l[0]) ** 2) + ((b_r[1] - b_l[1]) ** 2))
    # width_b = np.sqrt(((t_r[0] - t_l[0]) ** 2) + ((t_r[1] - t_l[1]) ** 2))
    width_a = np.sqrt(((rect[1, 0] - rect[0, 0]) ** 2) +
                      ((rect[1, 1] - rect[0, 1]) ** 2))
    width_b = np.sqrt(((rect[3, 0] - rect[2, 0]) ** 2) +
                      ((rect[3, 1] - rect[2, 1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((rect[0, 0] - rect[3, 0]) ** 2) +
                       ((rect[0, 1] - rect[3, 1]) ** 2))
    height_b = np.sqrt(((rect[1, 0] - rect[2, 0]) ** 2) +
                       ((rect[1, 1] - rect[2, 1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    rect = np.array([
        [rect[0, 0], rect[0, 1]],
        [rect[1, 0], rect[1, 1]],
        [rect[2, 0], rect[2, 1]],
        [rect[3, 0], rect[3, 1]]], dtype="float32")

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))

    # return the warped image
    return warped


def _line_intersection(x, y):
    """
    Calculates the intersection point of two lines, defined by the points
    (x1, y1) and (x2, y2) (first line), and
    (x3, y3) and (x4, y4) (second line).
    If the lines are parallel, (nan, nan) is returned.
    """
    slope_0 = (x[0] - x[1]) * (y[2] - y[3])
    slope_2 = (y[0] - y[1]) * (x[2] - x[3])
    if slope_0 == slope_2:
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xy_01 = x[0] * y[1] - y[0] * x[1]
        xy_23 = x[2] * y[3] - y[2] * x[3]
        denom = slope_0 - slope_2

        xis = (xy_01 * (x[2] - x[3]) - (x[0] - x[1]) * xy_23) / denom
        yis = (xy_01 * (y[2] - y[3]) - (y[0] - y[1]) * xy_23) / denom

    return (xis, yis)


def _simplify_polygon(in_poly,
                     length_cutoff=0.15,
                     maxiter=None,
                     segment_to_remove=None):
    """
    Removes segments from a (convex) polygon by continuing neighboring
    segments to a new point of intersection. Purpose is to approximate
    rounded polygons (quadrilaterals) with more sharp-cornered ones.
    """

    x_in = np.asarray(in_poly.exterior.coords)[:-1, 0]
    y_in = np.asarray(in_poly.exterior.coords)[:-1, 1]
    len_poly = len(x_in)
    niter = 0
    if segment_to_remove is not None:
        maxiter = 1
    while len_poly > 4:
        d_in = np.sqrt(np.ediff1d(x_in, to_end=x_in[0] - x_in[-1]) ** 2. +
                       np.ediff1d(y_in, to_end=y_in[0] - y_in[-1]) ** 2.)
        d_tot = np.sum(d_in)
        if segment_to_remove is not None:
            k = segment_to_remove
        else:
            k = np.argmin(d_in)
        if d_in[k] < length_cutoff * d_tot:
            ind = _generate_point_indices(k - 1, k + 1, len_poly)
            (xis, yis) = _line_intersection(x_in[ind], y_in[ind])
            x_in[k] = xis
            y_in[k] = yis
            x_in = np.delete(x_in, (k + 1) % len_poly)
            y_in = np.delete(y_in, (k + 1) % len_poly)
            len_poly = len(x_in)
            niter += 1
            if (maxiter is not None) and (niter >= maxiter):
                break
        else:
            break

    out_poly = Polygon([[ix, iy] for (ix, iy) in zip(x_in, y_in)])

    return out_poly


def _generate_point_indices(index_1, index_2, max_len):
    """
    Returns the four indices that give the end points of
    polygon segments corresponding to index_1 and index_2,
    modulo the number of points (max_len).
    """
    return np.array([index_1 % max_len,
                     (index_1 + 1) % max_len,
                     index_2 % max_len,
                     (index_2 + 1) % max_len])


def _generate_quad_corners(indices, x, y):
    """
    Returns the four intersection points from the
    segments defined by the x coordinates (x),
    y coordinates (y), and the indices.
    """
    (i, j, k, l) = indices

    def gpi(index_1, index_2):
        return _generate_point_indices(index_1, index_2, len(x))

    xis = np.empty(4)
    yis = np.empty(4)
    xis.fill(np.nan)
    yis.fill(np.nan)

    if j <= i or k <= j or l <= k:
        pass
    else:
        (xis[0], yis[0]) = _line_intersection(x[gpi(i, j)],
                                             y[gpi(i, j)])
        (xis[1], yis[1]) = _line_intersection(x[gpi(j, k)],
                                             y[gpi(j, k)])
        (xis[2], yis[2]) = _line_intersection(x[gpi(k, l)],
                                             y[gpi(k, l)])
        (xis[3], yis[3]) = _line_intersection(x[gpi(l, i)],
                                             y[gpi(l, i)])

    return (xis, yis)


def _generate_quad_candidates(in_poly):
    """
    Generates a list of bounding quadrilaterals for a polygon,
    using all possible combinations of four intersection points
    derived from four extended polygon segments.
    The number of combinations increases rapidly with the order
    of the polygon, so simplification should be applied first to
    remove very short segments from the polygon.
    """
    # make sure that the points are ordered
    (x_s, y_s) = _order_polygon_points(
        np.asarray(in_poly.exterior.coords)[:-1, 0],
        np.asarray(in_poly.exterior.coords)[:-1, 1])
    x_s_ave = np.average(x_s)
    y_s_ave = np.average(y_s)
    x_shrunk = x_s_ave + 0.9999 * (x_s - x_s_ave)
    y_shrunk = y_s_ave + 0.9999 * (y_s - y_s_ave)
    shrunk_poly = Polygon([[x, y] for (x, y) in
                           zip(x_shrunk, y_shrunk)])
    quads = []
    len_poly = len(x_s)

    for indices in product(range(len_poly), repeat=4):
        (xis, yis) = _generate_quad_corners(indices, x_s, y_s)
        if (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0:
            # no intersection point for some of the lines
            pass
        else:
            (xis, yis) = _order_polygon_points(xis, yis)
            enclose = True
            quad = Polygon([(xis[0], yis[0]),
                            (xis[1], yis[1]),
                            (xis[2], yis[2]),
                            (xis[3], yis[3])])
            if not quad.contains(shrunk_poly):
                enclose = False
            if enclose:
                quads.append(quad)
    return quads


def _get_bounding_quad(hull_poly):
    """
    Returns the minimum area quadrilateral that contains (bounds)
    the convex hull (openCV format) given as input.
    """
    simple_poly = _simplify_polygon(hull_poly)
    bounding_quads = _generate_quad_candidates(simple_poly)
    bquad_areas = np.zeros(len(bounding_quads))
    for iquad, bquad in enumerate(bounding_quads):
        bquad_areas[iquad] = bquad.area
    min_area_quad = bounding_quads[np.argmin(bquad_areas)]

    return min_area_quad


def _quad_corner_diff(hull_poly, bquad_poly, region_size=0.9):
    """
    Returns the difference between areas in the corners of a rounded
    corner and the aproximating sharp corner quadrilateral.
    region_size (param) determines the region around the corner where
    the comparison is done.
    """
    bquad_corners = np.zeros((4, 2))
    bquad_corners[:, 0] = np.asarray(bquad_poly.exterior.coords)[:-1, 0]
    bquad_corners[:, 1] = np.asarray(bquad_poly.exterior.coords)[:-1, 1]

    # The point inside the quadrilateral, region_size towards the quad center
    interior_points = np.zeros((4, 2))
    interior_points[:, 0] = np.average(bquad_corners[:, 0]) + \
        region_size * (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    interior_points[:, 1] = np.average(bquad_corners[:, 1]) + \
        region_size * (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))

    # The points p0 and p1 (at each corner) define the line whose intersections
    # with the quad together with the corner point define the triangular
    # area where the roundness of the convex hull in relation to the bounding
    # quadrilateral is evaluated.
    # The line (out of p0 and p1) is constructed such that it goes through the
    # "interior_point" and is orthogonal to the line going from the corner to
    # the center of the quad.
    p0_x = interior_points[:, 0] + \
        (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p1_x = interior_points[:, 0] - \
        (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p0_y = interior_points[:, 1] - \
        (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    p1_y = interior_points[:, 1] + \
        (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))

    corner_area_polys = []
    for i in range(len(interior_points[:, 0])):
        bline = LineString([(p0_x[i], p0_y[i]), (p1_x[i], p1_y[i])])
        corner_area_polys.append(Polygon(
            [bquad_poly.intersection(bline).coords[0],
             bquad_poly.intersection(bline).coords[1],
             (bquad_corners[i, 0], bquad_corners[i, 1])]))

    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        if (capoly.intersects(hull_poly)):
            hull_corner_area += capoly.intersection(hull_poly).area

    return 1. - hull_corner_area / quad_corner_area


def _convex_hull_polygon(contour):
    """
    Returns the convex hull of the given contour as a polygon.
    """
    hull = cv2.convexHull(contour)
    phull = Polygon([[x, y] for (x, y) in
                     zip(hull[:, :, 0], hull[:, :, 1])])
    return phull


def _polygon_form_factor(poly):
    """
    The ratio between the polygon area and circumference length,
    scaled by the length of the shortest segment.
    """
    # minimum side length
    d_0 = np.amin(np.sqrt(np.sum(np.diff(np.asarray(poly.exterior.coords),
                                         axis=0) ** 2., axis=1)))
    return poly.area / (poly.length * d_0)


def characterize_card_contour(card_contour,
                              max_segment_area,
                              image_area):
    """
    Calculates a bounding polygon for a contour, in addition
    to several charasteristic parameters.
    """
    phull = _convex_hull_polygon(card_contour)
    if (phull.area < 0.1 * max_segment_area or
            phull.area < image_area / 1000.):
        # break after card size range has been explored
        continue_segmentation = False
        is_card_candidate = False
        bounding_poly = None
        crop_factor = 1.
    else:
        continue_segmentation = True
        bounding_poly = _get_bounding_quad(phull)
        qc_diff = _quad_corner_diff(phull, bounding_poly)
        crop_factor = min(1., (1. - qc_diff * 22. / 100.))
        is_card_candidate = bool(
            0.1 * max_segment_area < bounding_poly.area <
            image_area * 0.99 and
            qc_diff < 0.35 and
            0.25 < _polygon_form_factor(bounding_poly) < 0.33)

    return (continue_segmentation,
            is_card_candidate,
            bounding_poly,
            crop_factor)