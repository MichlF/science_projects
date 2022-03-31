import numpy as np
from nilearn.image import load_img, new_img_like, resample_to_img, concat_imgs
from nilearn import plotting


def mask_merge(path_list_masks, resample_mask=False, interpol="nearest"):
    """Function to merge (i.e. build union of) two masks using nilearn functions (currently only supports merging two masks)

    Args:
        path_list_masks (_type_): path to mask(s)
        resample_mask (bool, optional): resample the intersected mask to the resolution of the first. Defaults to False.
        interpol (str, optional): interpolation method used for resampling. Defaults to "nearest".

    Raises:
        ValueError: assertion test for when too few masks are provided

    Returns:
        masks_list: returns the merged masks
    """    """"""

    print("INFO: Currently not working for more than 2 masks...")
    if len(path_list_masks) < 2:
        raise ValueError("Too few masks provided. At least 2 are necessary !")

    masks_list = []
    multiple_masks = False
    resample_reference = path_list_masks[0]

    for mask in path_list_masks:
        # Load masks
        mask1 = load_img(path_list_masks.pop(0))
        # Check if only a single mask is provided
        if path_list_masks:
            mask2 = load_img(path_list_masks.pop(0))
            multiple_masks = True

        # If necessary, resample
        if resample_mask:
            try:
                # Try loading the image
                mask_obj_resample = concat_imgs(resample_mask)
            except Exception as e:
                print('ERROR: ', e)
                # If error is raised, use first mask object as reference
                mask_obj_resample = load_img(resample_reference)
            print(f"Shapes: mask {mask1.shape}, functional data {mask_obj_resample.shape}")
            mask1 = resample_to_img(mask1, mask_obj_resample, interpolation=interpol)
            if multiple_masks:
                mask2 = resample_to_img(mask2, mask_obj_resample, interpolation=interpol)

        # Make masks boolean
        mask1_bool = mask1.get_fdata().astype(bool)
        # Merge mask by finding all voxels that are bool true in either mask
        if multiple_masks:
            mask2_bool = mask2.get_fdata().astype(bool)
            merged_mask = mask1_bool | mask2_bool
        else:
            merged_mask = mask1_bool
            print("WARNING: Merged mask is based on a single mask!")
        merged_mask = new_img_like(mask1, merged_mask)

        # Add to list
        masks_list.append(merged_mask)
        print(f"INFO: Merged mask has {np.sum(masks_list[-1].get_fdata().astype(bool))} voxels")

    # Only return a single obj if only two masks were merged
    if len(masks_list) == 1:
        masks_list = masks_list[0]

    return masks_list

# END