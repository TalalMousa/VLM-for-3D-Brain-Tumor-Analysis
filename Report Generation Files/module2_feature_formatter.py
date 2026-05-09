"""Compute structured clinical findings from labels and segmentation masks."""

from typing import Dict, Any, Optional
import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist


# Validate the class labels.
VALID_LABELS = {"No tumor", "Glioma", "Meningioma"}
LABEL_ALIASES = {
    "no_tumor": "No tumor",
    "no tumor": "No tumor",
    "no tumour": "No tumor",
    "healthy": "No tumor",
    "healthy brain": "No tumor",
    "healthy / no tumor": "No tumor",
    "glioma": "Glioma",
    "meningioma": "Meningioma",
}


def normalize_classification_label(classification_label: str) -> str:
    """Normalize classifier labels so healthy scans map cleanly to ``No tumor``."""
    if classification_label is None:
        raise ValueError("classification_label must not be None")

    stripped = classification_label.strip()
    if stripped in VALID_LABELS:
        return stripped
    return LABEL_ALIASES.get(stripped.lower(), stripped)


def compute_mask_volume_cm3(
    mask_data: np.ndarray,
    voxel_spacing: tuple,
) -> float:
    """
    Compute physical tumor volume in cubic centimetres.

    Parameters
    ----------
    mask_data : np.ndarray
        3D binary / multi-label segmentation mask (non-zero = tumor).
    voxel_spacing : tuple of float
        Voxel dimensions in millimetres, e.g. (1.0, 1.0, 1.0).
        Obtained from the NIfTI header via `nii.header.get_zooms()`.

    Returns
    -------
    float
        Tumor volume in cm^3.
    """
    # Count all non-zero mask voxels.
    voxel_count = int(np.count_nonzero(mask_data))
    # Compute the volume of one voxel in mm^3.
    voxel_vol_mm3 = float(np.prod(voxel_spacing[:3]))
    # Convert the total voxel volume from mm^3 to cm^3.
    volume_cm3 = (voxel_count * voxel_vol_mm3) / 1000.0

    return round(volume_cm3, 3)


def compute_max_diameter_mm(
    mask_data: np.ndarray,
    voxel_spacing: tuple,
) -> float:
    """Compute the maximum physical distance across the segmented mask."""
    coords = np.argwhere(mask_data > 0)
    if len(coords) < 2:
        return 0.0

    coords_mm = coords.astype(np.float32) * np.asarray(voxel_spacing[:3], dtype=np.float32)
    candidate_points = coords_mm

    if len(coords_mm) >= 4:
        try:
            hull = ConvexHull(coords_mm)
            candidate_points = coords_mm[hull.vertices]
        except QhullError:
            candidate_points = coords_mm

    if len(candidate_points) > 5000:
        step = int(np.ceil(len(candidate_points) / 5000))
        candidate_points = candidate_points[::step]

    return round(float(pdist(candidate_points).max()), 1)


def format_clinical_findings(
    classification_label: str,
    mask_volume_numpy: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a structured clinical-findings dictionary.

    Parameters
    ----------
    classification_label : str
        Predicted class label from the DenseNet classifier.
        Must be one of: "No tumor", "Glioma", "Meningioma".
    mask_volume_numpy : np.ndarray
        3D numpy array of the nnU-Net segmentation mask.
    voxel_spacing : tuple of float, optional
        Physical voxel dimensions in mm (default: 1x1x1 mm).
        Pass `nii.header.get_zooms()[:3]` for real data.
    patient_id : str, optional
        An optional patient / study identifier.

    Returns
    -------
    dict
        Structured dictionary with the following keys:
        - patient_id
        - classification_label
        - classification_confidence_note
        - tumor_present
        - tumor_voxel_count
        - voxel_spacing_mm
        - tumor_volume_cm3
        - tumor_location_centroid (i, j, k indices)
        - summary_text  (human-readable one-liner)

    Raises
    ------
    ValueError
        If `classification_label` is not in the allowed set.
    """

    # Validate the class label.

    classification_label = normalize_classification_label(classification_label)

    if classification_label not in VALID_LABELS:
        raise ValueError(
            f"Invalid classification label '{classification_label}'. "
            f"Must be one of {VALID_LABELS}."
        )

    # Compute the tumor volume.

    lesion_voxel_count = int(np.count_nonzero(mask_volume_numpy))
    lesion_volume_cm3 = compute_mask_volume_cm3(mask_volume_numpy, voxel_spacing)
    voxel_vol_mm3 = float(np.prod(voxel_spacing[:3]))
    tumor_core_mask = np.isin(mask_volume_numpy, [1, 3, 4])
    tumor_core_voxel_count = int(np.count_nonzero(tumor_core_mask))
    if tumor_core_voxel_count == 0 and lesion_voxel_count > 0:
        tumor_core_mask = mask_volume_numpy > 0
        tumor_core_voxel_count = lesion_voxel_count
    tumor_volume_cm3 = compute_mask_volume_cm3(tumor_core_mask, voxel_spacing)

    # Determine if a tumor is present.

    tumor_present = classification_label != "No tumor" and lesion_voxel_count > 0

    # Compute detailed morphometry when a tumor is present.

    if tumor_present:
        coords = np.argwhere(mask_volume_numpy > 0)  # Nx3 (i, j, k)
        # Compute the tumor centroid.
        centroid = tuple(coords.mean(axis=0).astype(int).tolist())
        # Compute the 3D bounding box.
        bb_min = coords.min(axis=0).tolist()  # [i_min, j_min, k_min]
        bb_max = coords.max(axis=0).tolist()  # [i_max, j_max, k_max]
        bb_size_voxels = [mx - mn + 1 for mn, mx in zip(bb_min, bb_max)]
        bb_size_mm = [round(s * sp, 1) for s, sp in zip(bb_size_voxels, voxel_spacing[:3])]
        # Compute the maximum physical diameter in mm.
        max_diameter_mm = compute_max_diameter_mm(mask_volume_numpy > 0, voxel_spacing)
        # Compute the axial span.
        axial_min = int(coords[:, 2].min())
        axial_max = int(coords[:, 2].max())
        axial_span_slices = axial_max - axial_min + 1
        axial_span_mm = round(axial_span_slices * float(voxel_spacing[2]), 1)
        # Estimate the surface area using voxel faces.
        # Count exposed faces of tumor voxels.
        binary_mask = (mask_volume_numpy > 0).astype(np.uint8)
        surface_faces = 0
        for axis in range(3):
            shifted = np.roll(binary_mask, 1, axis=axis)
            # Boundary faces: tumor voxel adjacent to non-tumor
            surface_faces += int(np.sum(binary_mask != shifted))
        # Each face has area = product of the two perpendicular spacings
        face_areas = [
            float(voxel_spacing[1]) * float(voxel_spacing[2]),  # Faces perpendicular to axis 0.
            float(voxel_spacing[0]) * float(voxel_spacing[2]),  # Faces perpendicular to axis 1.
            float(voxel_spacing[0]) * float(voxel_spacing[1]),  # Faces perpendicular to axis 2.
        ]
        avg_face_area = sum(face_areas) / 3.0
        surface_area_mm2 = round(surface_faces * avg_face_area, 1)
        surface_area_cm2 = round(surface_area_mm2 / 100.0, 2)
        # Compute sphericity, where 1.0 is a perfect sphere.
        volume_mm3 = lesion_voxel_count * voxel_vol_mm3
        if surface_area_mm2 > 0:
            sphericity = round(
                (np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / surface_area_mm2, 3
            )
        else:
            sphericity = 0.0
        # Compute BraTS sub-region volumes.
        unique_labels = np.unique(mask_volume_numpy[mask_volume_numpy > 0]).astype(int).tolist()
        subregion_volumes = {}
        subregion_label_names = {
            1: "Necrotic Core (NCR)",
            2: "Peritumoral Edema (ED)",
            3: "Enhancing Tumor (ET)",
            4: "Enhancing Tumor (ET)",
        }
        for lbl in unique_labels:
            count = int(np.count_nonzero(mask_volume_numpy == lbl))
            vol = round((count * voxel_vol_mm3) / 1000.0, 3)
            name = subregion_label_names.get(lbl, f"Label {lbl}")
            subregion_volumes[name] = {
                "label_value": lbl,
                "voxel_count": count,
                "volume_cm3": vol,
                "percentage": round(count / lesion_voxel_count * 100, 1),
            }
        # Determine tumor hemisphere.
        mid_sagittal = mask_volume_numpy.shape[0] // 2
        laterality_mask = tumor_core_mask if tumor_core_voxel_count > 0 else (mask_volume_numpy > 0)
        left_count = int(np.count_nonzero(laterality_mask[:mid_sagittal, :, :]))
        right_count = int(np.count_nonzero(laterality_mask[mid_sagittal:, :, :]))
        total_laterality = left_count + right_count
        if left_count > 0 and right_count > 0:
            left_fraction = left_count / total_laterality
            right_fraction = right_count / total_laterality
            if left_fraction >= 0.2 and right_fraction >= 0.2:
                dominant_side = "Left-dominant" if left_count > right_count else "Right-dominant"
                hemisphere = f"Bilateral ({dominant_side})"
            elif left_count > right_count:
                hemisphere = "Left hemisphere"
            else:
                hemisphere = "Right hemisphere"
        elif left_count > 0:
            hemisphere = "Left hemisphere"
        elif right_count > 0:
            hemisphere = "Right hemisphere"
        else:
            hemisphere = "Indeterminate"

    else:
        centroid = None
        bb_min = bb_max = bb_size_voxels = bb_size_mm = None
        max_diameter_mm = 0.0
        axial_min = axial_max = axial_span_slices = 0
        axial_span_mm = 0.0
        surface_area_cm2 = 0.0
        sphericity = 0.0
        subregion_volumes = {}
        hemisphere = "N/A"

    # Build the clinical findings dictionary.

    findings: Dict[str, Any] = {
        "patient_id": patient_id or "UNKNOWN",
        "classification_label": classification_label,
        "classification_confidence_note": (
            "Label produced by DenseNet-based classifier. "
            "Confirm with histopathological examination."
        ),
        "tumor_present": tumor_present,
        # Volume metrics.
        "tumor_voxel_count": tumor_core_voxel_count,
        "segmented_lesion_voxel_count": lesion_voxel_count,
        "voxel_spacing_mm": tuple(float(s) for s in voxel_spacing[:3]),
        "tumor_volume_cm3": tumor_volume_cm3,
        "segmented_lesion_volume_cm3": lesion_volume_cm3,
        # Spatial location.
        "tumor_location_centroid": centroid,
        "hemisphere": hemisphere,
        # Bounding box and diameter.
        "bounding_box_min_ijk": bb_min,
        "bounding_box_max_ijk": bb_max,
        "bounding_box_size_mm": bb_size_mm,
        "max_diameter_mm": max_diameter_mm,
        # Axial extent.
        "axial_slice_range": (axial_min, axial_max) if tumor_present else None,
        "axial_span_slices": axial_span_slices,
        "axial_span_mm": axial_span_mm,
        # Surface and shape.
        "surface_area_cm2": surface_area_cm2,
        "sphericity": sphericity,
        # Sub-region breakdown.
        "subregion_volumes": subregion_volumes,
    }

    # Build the human-readable summary.

    if not tumor_present:
        summary = (
            f"Patient {findings['patient_id']}: No intracranial mass detected. "
            f"Classification = '{classification_label}'."
        )
    else:
        subregion_str = ", ".join(
            f"{name}: {info['volume_cm3']} cm^3 ({info['percentage']}%)"
            for name, info in subregion_volumes.items()
        )
        summary = (
            f"Patient {findings['patient_id']}: {classification_label} detected. "
            f"Tumor core volume = {tumor_volume_cm3} cm^3, "
            f"segmented lesion volume = {lesion_volume_cm3} cm^3, "
            f"max diameter = {max_diameter_mm} mm, "
            f"surface area = {surface_area_cm2} cm^2, "
            f"sphericity = {sphericity}, "
            f"location = {hemisphere}, "
            f"axial span = {axial_span_mm} mm ({axial_span_slices} slices). "
            f"Sub-regions: [{subregion_str}]."
        )

    findings["summary_text"] = summary
    print(f"[Module 2] {summary}")

    return findings


# ---------------------------------------------------------------------------
# Advanced metrics extraction (strict JSON format)
# ---------------------------------------------------------------------------


def _per_class_volume_cm3(
    mask: np.ndarray,
    label: int,
    voxel_vol_mm3: float,
) -> float:
    """Volume of a single label class in cm³."""
    count = int(np.count_nonzero(mask == label))
    return round((count * voxel_vol_mm3) / 1000.0, 2)


def extract_advanced_metrics(
    mask_data: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Extract advanced tumor metrics in the strict output format.

    Returns a dict with all values rounded to 2 decimals and tagged as
    ``"type": "segmentation_estimate"``.

    Parameters
    ----------
    mask_data : np.ndarray
        3D segmentation mask (labels: 0=bg, 1=NCR, 2=ED, 3=ET).
    voxel_spacing : tuple of float
        Voxel dimensions in mm.

    Returns
    -------
    dict
        Strict-format tumor_metrics dict ready for pipeline JSON.
    """
    voxel_vol_mm3 = float(np.prod(voxel_spacing[:3]))

    # Per-class volumes
    ncr_vol = _per_class_volume_cm3(mask_data, 1, voxel_vol_mm3)
    ed_vol = _per_class_volume_cm3(mask_data, 2, voxel_vol_mm3)
    et_vol = _per_class_volume_cm3(mask_data, 3, voxel_vol_mm3)

    # Composite volumes
    tumor_core_vol = round(ncr_vol + et_vol, 2)  # NCR + ET
    total_lesion_vol = round(ncr_vol + ed_vol + et_vol, 2)

    # Geometry
    max_diam = compute_max_diameter_mm(mask_data > 0, voxel_spacing)

    coords = np.argwhere(mask_data > 0)
    if len(coords) >= 2:
        bb_min = coords.min(axis=0).tolist()
        bb_max = coords.max(axis=0).tolist()
        bb_size_mm = [round((mx - mn + 1) * sp, 2) for mn, mx, sp in zip(bb_min, bb_max, voxel_spacing[:3])]

        axial_min = int(coords[:, 2].min())
        axial_max = int(coords[:, 2].max())
        axial_span_mm = round((axial_max - axial_min + 1) * float(voxel_spacing[2]), 2)

        # Surface area
        binary = (mask_data > 0).astype(np.uint8)
        surface_faces = 0
        for axis in range(3):
            shifted = np.roll(binary, 1, axis=axis)
            surface_faces += int(np.sum(binary != shifted))
        face_areas = [
            float(voxel_spacing[1]) * float(voxel_spacing[2]),
            float(voxel_spacing[0]) * float(voxel_spacing[2]),
            float(voxel_spacing[0]) * float(voxel_spacing[1]),
        ]
        avg_face_area = sum(face_areas) / 3.0
        surface_area_cm2 = round(surface_faces * avg_face_area / 100.0, 2)

        # Sphericity
        volume_mm3 = int(np.count_nonzero(mask_data)) * voxel_vol_mm3
        surface_area_mm2 = surface_faces * avg_face_area
        if surface_area_mm2 > 0:
            sphericity = round(
                (np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / surface_area_mm2, 2
            )
        else:
            sphericity = 0.0
    else:
        bb_size_mm = [0.0, 0.0, 0.0]
        axial_span_mm = 0.0
        surface_area_cm2 = 0.0
        sphericity = 0.0

    metrics = {
        "total_volume_cm3": total_lesion_vol,
        "tumor_core_volume_cm3": tumor_core_vol,
        "ncr_volume_cm3": ncr_vol,
        "edema_volume_cm3": ed_vol,
        "enhancing_volume_cm3": et_vol,
        "max_diameter_mm": round(max_diam, 2),
        "axial_span_mm": axial_span_mm,
        "bounding_box_mm": bb_size_mm,
        "surface_area_cm2": surface_area_cm2,
        "sphericity": sphericity,
        "type": "segmentation_estimate",
    }

    # Consistency checks
    has_tumor = total_lesion_vol > 0
    if has_tumor:
        assert total_lesion_vol > 0, "Volume must be > 0 when tumor is present"
        # Sanity: diameter should be in a reasonable range relative to volume
        expected_min_diam = (6 * total_lesion_vol * 1000 / np.pi) ** (1 / 3) * 0.3
        if max_diam > 0 and max_diam < expected_min_diam * 0.01:
            print(f"[Module 2] WARNING: Max diameter ({max_diam} mm) seems very small "
                  f"for volume ({total_lesion_vol} cm³)")

    # Assert no NaN
    for key, val in metrics.items():
        if isinstance(val, float):
            assert not np.isnan(val), f"NaN found in metric: {key}"

    print(f"[Module 2] Advanced metrics: total={total_lesion_vol} cm³, "
          f"core={tumor_core_vol} cm³, diam={max_diam} mm, sphericity={sphericity}")
    return metrics
