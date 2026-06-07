# analyze_rock_structures.py

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from scipy import ndimage as ndi
from skimage import measure, morphology


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path("../poros/poros/")
OUTPUT_CSV = Path("descriptor.csv")

PORE_RGB = np.array([0, 255, 0], dtype=np.uint8)

# Use 0 for a perfect green/black mask.
# Use 10, 20, or 30 if there is color variation due to compression.
GREEN_TOLERANCE = 0

LACUNARITY_BOX_SIZE = 32

# Important correction:
# very small clusters distort perimeter/circularity measurements.
MIN_CLUSTER_AREA_FOR_SHAPE = 20

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def pore_mask_from_green(rgb_array, tolerance=0):
    """
    True = pore
    False = matrix
    """
    rgb = rgb_array.astype(np.int16)
    target = PORE_RGB.astype(np.int16)
    diff = np.abs(rgb - target)

    if tolerance == 0:
        return np.all(diff == 0, axis=2)

    return np.all(diff <= tolerance, axis=2)


# ============================================================
# SPATIAL DESCRIPTORS
# ============================================================

def calculate_perimeter(mask):
    """
    Total perimeter of the pore phase.
    Uses the Crofton perimeter, which is more stable than a simple contour.
    """
    if mask.sum() == 0:
        return 0.0

    return float(measure.perimeter_crofton(mask, directions=4))


def calculate_lacunarity(mask, box_size=32):
    h, w = mask.shape
    masses = []

    for y in range(0, h - box_size + 1, box_size):
        for x in range(0, w - box_size + 1, box_size):
            box = mask[y:y + box_size, x:x + box_size]
            masses.append(box.sum())

    masses = np.array(masses, dtype=np.float64)

    if len(masses) == 0:
        return 0.0

    mean_mass = masses.mean()

    if mean_mass == 0:
        return 0.0

    return float(masses.var() / (mean_mass ** 2) + 1.0)


def boxcount(mask, box_size):
    h, w = mask.shape

    h_crop = h - (h % box_size)
    w_crop = w - (w % box_size)

    if h_crop == 0 or w_crop == 0:
        return 0

    cropped = mask[:h_crop, :w_crop]

    reshaped = cropped.reshape(
        h_crop // box_size,
        box_size,
        w_crop // box_size,
        box_size
    )

    boxes = reshaped.any(axis=(1, 3))
    return int(boxes.sum())


def calculate_fractal_dimension(mask):
    """
    Approximate fractal dimension computed by box-counting.
    """
    if mask.sum() == 0:
        return 0.0

    min_dim = min(mask.shape)

    sizes = []
    size = 2

    while size <= min_dim // 2:
        sizes.append(size)
        size *= 2

    if len(sizes) < 2:
        return 0.0

    counts = []

    for s in sizes:
        c = boxcount(mask, s)
        counts.append(c if c > 0 else np.nan)

    sizes = np.array(sizes, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)

    valid = np.isfinite(counts) & (counts > 0)

    if valid.sum() < 2:
        return 0.0

    log_sizes = np.log(1.0 / sizes[valid])
    log_counts = np.log(counts[valid])

    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return float(slope)


def calculate_skeleton_descriptors(mask):
    if mask.sum() == 0:
        return 0, 0.0

    skeleton = morphology.skeletonize(mask)
    skeleton_length = int(skeleton.sum())
    skeleton_density = safe_divide(skeleton_length, mask.size)

    return skeleton_length, skeleton_density


def calculate_euler_number(mask):
    if mask.sum() == 0:
        return 0

    return int(measure.euler_number(mask, connectivity=2))


# ============================================================
# CLUSTERS AND CIRCULARITY CORRECTION
# ============================================================

def corrected_circularity(area, perimeter):
    """
    Classical circularity:
        C = 4*pi*A / P²

    In small digital objects, perimeter errors may generate C > 1.
    Here:
    - returns 0 if the perimeter is invalid;
    - clips the value to the interval [0, 1].
    """
    if perimeter <= 0 or area <= 0:
        return 0.0

    c = 4.0 * math.pi * float(area) / (float(perimeter) ** 2)

    if not np.isfinite(c):
        return 0.0

    return float(np.clip(c, 0.0, 1.0))


def calculate_cluster_properties(mask):
    """
    Computes properties of pore clusters.

    The interpretive circularity uses only clusters with area >=
    MIN_CLUSTER_AREA_FOR_SHAPE, because very small clusters distort
    perimeter, circularity, aspect ratio, and eccentricity.
    """
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n_clusters = ndi.label(mask, structure=structure)

    if n_clusters == 0:
        return {
            "labels": labels,
            "n_pore_clusters": 0,
            "cluster_areas": np.array([], dtype=float),
            "largest_cluster_area_px": 0.0,
            "largest_cluster_fraction_of_pores": 0.0,
            "mean_cluster_area_px": 0.0,
            "median_cluster_area_px": 0.0,
            "p90_cluster_area_px": 0.0,
            "p95_cluster_area_px": 0.0,
            "cluster_area_cv": 0.0,
            "n_valid_shape_clusters": 0,
            "mean_cluster_perimeter_px": 0.0,
            "mean_circularity": 0.0,
            "mean_circularity_valid_clusters": 0.0,
            "area_weighted_circularity": 0.0,
            "mean_aspect_ratio": 0.0,
            "mean_solidity": 0.0,
            "mean_eccentricity": 0.0,
        }

    cluster_areas = np.bincount(labels.ravel())[1:].astype(float)

    largest_cluster_area = float(cluster_areas.max())
    total_pore_area = float(mask.sum())

    props = measure.regionprops(labels)

    valid_areas = []
    valid_perimeters = []
    valid_circularities = []
    valid_aspect_ratios = []
    valid_solidities = []
    valid_eccentricities = []

    all_circularities = []

    for region in props:
        area = float(region.area)

        # Crofton is more robust for digital objects.
        perimeter = float(region.perimeter_crofton)

        circ = corrected_circularity(area, perimeter)
        all_circularities.append(circ)

        if area < MIN_CLUSTER_AREA_FOR_SHAPE:
            continue

        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc

        if min(height, width) > 0:
            aspect_ratio = max(height, width) / min(height, width)
        else:
            aspect_ratio = 0.0

        valid_areas.append(area)
        valid_perimeters.append(perimeter)
        valid_circularities.append(circ)
        valid_aspect_ratios.append(aspect_ratio)
        valid_solidities.append(float(region.solidity))
        valid_eccentricities.append(float(region.eccentricity))

    mean_area = float(cluster_areas.mean())
    std_area = float(cluster_areas.std())
    cluster_area_cv = safe_divide(std_area, mean_area)

    valid_areas = np.array(valid_areas, dtype=float)
    valid_circularities = np.array(valid_circularities, dtype=float)

    if len(valid_areas) > 0:
        area_weighted_circularity = safe_divide(
            np.sum(valid_circularities * valid_areas),
            np.sum(valid_areas)
        )
    else:
        area_weighted_circularity = 0.0

    return {
        "labels": labels,
        "n_pore_clusters": int(n_clusters),
        "cluster_areas": cluster_areas,
        "largest_cluster_area_px": largest_cluster_area,
        "largest_cluster_fraction_of_pores": safe_divide(
            largest_cluster_area,
            total_pore_area
        ),
        "mean_cluster_area_px": mean_area,
        "median_cluster_area_px": float(np.median(cluster_areas)),
        "p90_cluster_area_px": float(np.percentile(cluster_areas, 90)),
        "p95_cluster_area_px": float(np.percentile(cluster_areas, 95)),
        "cluster_area_cv": cluster_area_cv,

        "n_valid_shape_clusters": int(len(valid_areas)),
        "mean_cluster_perimeter_px": (
            float(np.mean(valid_perimeters)) if len(valid_perimeters) > 0 else 0.0
        ),

        # Kept for compatibility, but now corrected to 0–1.
        "mean_circularity": (
            float(np.mean(all_circularities)) if len(all_circularities) > 0 else 0.0
        ),

        # More interpretive: only sufficiently large clusters.
        "mean_circularity_valid_clusters": (
            float(np.mean(valid_circularities)) if len(valid_circularities) > 0 else 0.0
        ),

        # More geologically robust: larger clusters receive more weight.
        "area_weighted_circularity": float(area_weighted_circularity),

        "mean_aspect_ratio": (
            float(np.mean(valid_aspect_ratios)) if len(valid_aspect_ratios) > 0 else 0.0
        ),
        "mean_solidity": (
            float(np.mean(valid_solidities)) if len(valid_solidities) > 0 else 0.0
        ),
        "mean_eccentricity": (
            float(np.mean(valid_eccentricities)) if len(valid_eccentricities) > 0 else 0.0
        ),
    }


# ============================================================
# INTERPRETATIONS
# ============================================================

def porosity_interpretation(porosity):
    if porosity < 0.05:
        return "very low porosity"
    elif porosity < 0.12:
        return "low to moderate porosity"
    elif porosity < 0.25:
        return "moderate porosity"
    elif porosity < 0.40:
        return "very high porosity"
    else:
        return "extremely high porosity"


def connectivity_interpretation(largest_cluster_fraction_of_pores):
    f = largest_cluster_fraction_of_pores

    if f < 0.20:
        return "highly fragmented and weakly connected pores"
    elif f < 0.50:
        return "intermediate connectivity with a relevant main cluster"
    elif f < 0.75:
        return "well-connected pore network dominated by a main cluster"
    else:
        return "strongly connected pore network dominated by a single cluster"


def heterogeneity_interpretation(cluster_area_cv, lacunarity):
    if cluster_area_cv < 2.0 and lacunarity < 2.0:
        return "relatively homogeneous texture"
    elif cluster_area_cv < 5.0 and lacunarity < 4.0:
        return "heterogeneous texture with moderate pore-size variation"
    else:
        return "highly heterogeneous texture with mixed micropores and large aggregates"


def circularity_interpretation(area_weighted_circularity):
    """
    Low circularity: elongated/irregular pores.
    High circularity: more rounded/compact pores.
    """
    c = area_weighted_circularity

    if c < 0.25:
        return "dominant pores are highly irregular or elongated"
    elif c < 0.50:
        return "dominant pores are irregular, with complex boundaries"
    elif c < 0.75:
        return "dominant pores are moderately compact"
    else:
        return "dominant pores are more rounded or compact"


# ============================================================
# MARION-LIKE
# ============================================================

def marion_ratio_distance(porosity):
    return abs(float(porosity) - 0.10)


def marion_like_class(porosity):
    d = marion_ratio_distance(porosity)

    if d <= 0.01:
        return "very close to the 1/10 regime"
    elif d <= 0.03:
        return "close to the 1/10 regime"
    elif d <= 0.08:
        return "moderately distant from the 1/10 regime"
    else:
        return "distant from the 1/10 regime"


def structural_descriptor(row):
    porosity = row["porosity_2d_fraction"]
    main_cluster = row["largest_cluster_fraction_of_pores"]
    lacunarity = row["lacunarity_box32"]
    fractal = row["fractal_dimension_boxcount"]
    marion_class = row["marion_like_class"]
    circ_interp = row["circularity_interpretation"]

    if porosity < 0.12 and main_cluster < 0.35:
        return (
            "compact microstructure, low to moderate porosity, "
            "fragmented pores and limited connectivity; "
            f"{marion_class}; {circ_interp}"
        )

    elif 0.12 <= porosity < 0.35 and main_cluster < 0.65:
        return (
            "heterogeneous microstructure, high porosity, "
            "presence of channels and partially connected porous aggregates; "
            f"{marion_class}; {circ_interp}"
        )

    elif porosity >= 0.35 and main_cluster >= 0.75:
        return (
            "dominant pore network, high spatial continuity of pores, "
            "open structure with strong connectivity; "
            f"{marion_class}; {circ_interp}"
        )

    elif porosity >= 0.35 and main_cluster < 0.75:
        return (
            "very high porosity, but with irregular connectivity; "
            "the structure contains large porous domains without absolute dominance "
            f"of a single cluster; {marion_class}; {circ_interp}"
        )

    elif lacunarity > 5.0 and fractal > 1.6:
        return (
            "highly heterogeneous and geometrically complex structure, "
            "with strong spatial variation between micropores and large aggregates; "
            f"{marion_class}; {circ_interp}"
        )

    else:
        return (
            "mixed microstructure, with significant porosity and irregular connectivity; "
            f"{marion_class}; {circ_interp}"
        )


# ============================================================
# SINGLE-IMAGE ANALYSIS
# ============================================================

def analyze_image(path):
    rgb = load_image_rgb(path)
    mask = pore_mask_from_green(rgb, tolerance=GREEN_TOLERANCE)

    height, width = mask.shape
    total_area = int(mask.size)
    pore_area = int(mask.sum())
    matrix_area = int(total_area - pore_area)

    porosity = safe_divide(pore_area, total_area)

    clusters = calculate_cluster_properties(mask)

    largest_cluster_fraction_of_image = safe_divide(
        clusters["largest_cluster_area_px"],
        total_area
    )

    total_pore_perimeter = calculate_perimeter(mask)

    specific_perimeter_per_image_area = safe_divide(
        total_pore_perimeter,
        total_area
    )

    specific_perimeter_per_pore_area = safe_divide(
        total_pore_perimeter,
        pore_area
    )

    euler = calculate_euler_number(mask)
    skeleton_length, skeleton_density = calculate_skeleton_descriptors(mask)

    lacunarity = calculate_lacunarity(mask, box_size=LACUNARITY_BOX_SIZE)
    fractal_dimension = calculate_fractal_dimension(mask)

    row = {
        "image": path.name,
        "width_px": int(width),
        "height_px": int(height),
        "total_area_px": int(total_area),
        "pore_area_px": int(pore_area),
        "matrix_area_px": int(matrix_area),
        "porosity_2d_fraction": float(porosity),
        "porosity_2d_percent": float(porosity * 100.0),

        "n_pore_clusters": clusters["n_pore_clusters"],
        "largest_cluster_area_px": clusters["largest_cluster_area_px"],
        "largest_cluster_fraction_of_image": largest_cluster_fraction_of_image,
        "largest_cluster_fraction_of_pores": clusters[
            "largest_cluster_fraction_of_pores"
        ],

        "mean_cluster_area_px": clusters["mean_cluster_area_px"],
        "median_cluster_area_px": clusters["median_cluster_area_px"],
        "p90_cluster_area_px": clusters["p90_cluster_area_px"],
        "p95_cluster_area_px": clusters["p95_cluster_area_px"],
        "cluster_area_cv": clusters["cluster_area_cv"],

        "n_valid_shape_clusters": clusters["n_valid_shape_clusters"],
        "total_pore_perimeter_px": total_pore_perimeter,
        "specific_perimeter_per_image_area": specific_perimeter_per_image_area,
        "specific_perimeter_per_pore_area": specific_perimeter_per_pore_area,
        "mean_cluster_perimeter_px": clusters["mean_cluster_perimeter_px"],

        "mean_circularity": clusters["mean_circularity"],
        "mean_circularity_valid_clusters": clusters[
            "mean_circularity_valid_clusters"
        ],
        "area_weighted_circularity": clusters["area_weighted_circularity"],

        "mean_aspect_ratio": clusters["mean_aspect_ratio"],
        "mean_solidity": clusters["mean_solidity"],
        "mean_eccentricity": clusters["mean_eccentricity"],

        "euler_number": euler,
        "skeleton_length_px": skeleton_length,
        "skeleton_density": skeleton_density,

        "lacunarity_box32": lacunarity,
        "fractal_dimension_boxcount": fractal_dimension,

        "porosity_interpretation": porosity_interpretation(porosity),
        "connectivity_interpretation": connectivity_interpretation(
            clusters["largest_cluster_fraction_of_pores"]
        ),
        "heterogeneity_interpretation": heterogeneity_interpretation(
            clusters["cluster_area_cv"],
            lacunarity
        ),
        "circularity_interpretation": circularity_interpretation(
            clusters["area_weighted_circularity"]
        ),
    }

    row["marion_ratio_distance"] = marion_ratio_distance(porosity)
    row["marion_like_class"] = marion_like_class(porosity)
    row["structural_descriptor"] = structural_descriptor(row)

    return row


# ============================================================
# EXECUTION
# ============================================================

def find_images(input_dir):
    if not input_dir.exists():
        raise FileNotFoundError(
            f"The folder '{input_dir}' does not exist. "
            "Create the folder and place your masks inside it."
        )

    images = []

    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


def main():
    warnings.filterwarnings("ignore")

    images = find_images(INPUT_DIR)

    if len(images) == 0:
        raise RuntimeError(
            f"No images found in '{INPUT_DIR}'. "
            "Place .png, .jpg, .tif, .bmp, or .webp files in the folder."
        )

    rows = []

    print(f"Found {len(images)} images in '{INPUT_DIR}'.")

    for path in images:
        print(f"Analyzing: {path.name}")
        rows.append(analyze_image(path))

    df = pd.DataFrame(rows)

    preferred_order = [
        "image",
        "width_px",
        "height_px",
        "total_area_px",
        "pore_area_px",
        "matrix_area_px",
        "porosity_2d_fraction",
        "porosity_2d_percent",

        "n_pore_clusters",
        "largest_cluster_area_px",
        "largest_cluster_fraction_of_image",
        "largest_cluster_fraction_of_pores",

        "mean_cluster_area_px",
        "median_cluster_area_px",
        "p90_cluster_area_px",
        "p95_cluster_area_px",
        "cluster_area_cv",

        "n_valid_shape_clusters",
        "total_pore_perimeter_px",
        "specific_perimeter_per_image_area",
        "specific_perimeter_per_pore_area",
        "mean_cluster_perimeter_px",

        "mean_circularity",
        "mean_circularity_valid_clusters",
        "area_weighted_circularity",
        "mean_aspect_ratio",
        "mean_solidity",
        "mean_eccentricity",

        "euler_number",
        "skeleton_length_px",
        "skeleton_density",
        "lacunarity_box32",
        "fractal_dimension_boxcount",

        "porosity_interpretation",
        "connectivity_interpretation",
        "heterogeneity_interpretation",
        "circularity_interpretation",

        "marion_ratio_distance",
        "marion_like_class",
        "structural_descriptor",
    ]

    df = df[preferred_order]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"File saved: {OUTPUT_CSV.resolve()}")
    print()
    print(df[[
        "image",
        "porosity_2d_percent",
        "largest_cluster_fraction_of_pores",
        "area_weighted_circularity",
        "marion_ratio_distance",
        "marion_like_class",
        "structural_descriptor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
