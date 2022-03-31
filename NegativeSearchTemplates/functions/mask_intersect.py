import numpy as np
from nilearn.masking import intersect_masks
from nilearn.image import load_img, new_img_like, resample_to_img
from nilearn import plotting


def mask_intersect(path_list_masks, resample_mask=False, interpol="nearest"):
    """Function to intersect two masks using nilearn functions (currently only supports intersecting two masks)

    Args:
        path_list_masks (_type_): path to mask(s)
        resample_mask (bool, optional): resample the intersected mask to the resolution of the first. Defaults to False.
        interpol (str, optional): interpolation method used for resampling. Defaults to "nearest".

    Raises:
        ValueError: assertion test for when too few masks are provided

    Returns:
        masks_list: returns the intersected masks
    """

    print("INFO: Currently not working for more than 2 masks...")
    if len(path_list_masks) < 2:
        raise ValueError("Too few masks provided. At least 2 are necessary !")

    resample_reference = path_list_masks[0]

    masks_list = []
    for _mask in path_list_masks:
        # Load masks
        mask1 = load_img(path_list_masks.pop(0))
        mask2 = load_img(path_list_masks.pop(0))

        # If necessary, resample
        if resample_mask:
            try:
                # Try loading the image
                mask_obj_resample = load_img(resample_mask)
            except Exception as e:
                print('ERROR: ', e)
                # If error is raised, use first mask object as reference
                mask_obj_resample = load_img(resample_reference)

            mask1 = resample_to_img(mask_obj_resample, mask1, interpolation=interpol)
            mask2 = resample_to_img(mask_obj_resample, mask2, interpolation=interpol)

        # Make masks boolean
        mask1_bool = mask1.get_fdata().astype(bool)
        mask1 = new_img_like(mask1, mask1_bool)
        mask2_bool = mask2.get_fdata().astype(bool)
        mask2 = new_img_like(mask2, mask2_bool)

        # Intersect masks
        masks_list.append(intersect_masks([mask1, mask2]))
        print(f"intersected mask has {np.sum(masks_list[-1].get_fdata().astype(bool))} voxels")

    # Only return a single obj if only two masks were merged
    if len(masks_list) == 1:
        masks_list = masks_list[0]

    return masks_list

# END