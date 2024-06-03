import torch

def select_elements(index, current_offsets_tensor, mla_index_tensor):
    """
    Selects elements from current_offsets_tensor and mla_index_tensor based on the given index.
    
    Args:
        index (int): The index to select.
        current_offsets_tensor (torch.Tensor): Tensor containing current offsets.
        mla_index_tensor (torch.Tensor): Tensor containing MLA indices.

    Returns:
        Tuple: A tuple containing the selected elements from current_offsets_tensor and mla_index_tensor.
    """
    current_offsets_selected = current_offsets_tensor[index]
    mla_index_selected = mla_index_tensor[index]

    return current_offsets_selected, mla_index_selected


def instance_iterable_inputs_for_image_generation(self, mla_indices, offsets, volume_in, intensity):
    def get_input(i):
        ml_jj_idx, ml_ii_idx = mla_indices[i]
        current_offset = offsets[i]
        img_list = self.generate_images(volume_in, current_offset, intensity, mla_index=(ml_jj_idx, ml_ii_idx))      
        return img_list
    return get_input


def process_row(self, ml_ii_idx, row_iterable, n_ml_half, odd_mla_shift, volume_in, intensity):
    n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
    n_micro_lenses = self.optical_info['n_micro_lenses']
    full_img_row_list = [None] * 5
    for ml_jj_idx, ml_jj in enumerate(range(-n_ml_half, n_ml_half+odd_mla_shift)):
        current_offset = self._calculate_current_offset(
            row_iterable[ml_ii_idx], ml_jj, n_voxels_per_ml, n_micro_lenses)
        img_list = self.generate_images(volume_in, current_offset,
                        intensity, mla_index=(ml_jj_idx, ml_ii_idx))
        full_img_row_list = self._concatenate_images(
            full_img_row_list, img_list, axis=0)
    return full_img_row_list


def make_process_row_method(instance):
    def process_row(ml_ii_idx, row_iterable, n_ml_half, odd_mla_shift, volume_in, intensity):
        full_img_row_list = [None] * 5
        for ml_jj_idx, ml_jj in enumerate(range(-n_ml_half, n_ml_half+odd_mla_shift)):
            current_offset = instance._calculate_current_offset(
                row_iterable[ml_ii_idx], ml_jj, instance.n_voxels_per_ml, instance.n_micro_lenses)
            img_list = instance.generate_images(volume_in, current_offset,
                            intensity, mla_index=(ml_jj_idx, ml_ii_idx))
            full_img_row_list = instance._concatenate_images(
                full_img_row_list, img_list, axis=0)
        return full_img_row_list
    return process_row


#### Attempt to parallelize the image generation process

# Initialize tensors to store inputs and outputs
num_pix = self.optical_info['pixels_per_ml']
full_img_list_array = torch.empty((2, n_micro_lenses * num_pix, n_micro_lenses * num_pix))
ml_ii_tensor = torch.arange(-n_ml_half, n_ml_half + odd_mla_shift)
ml_jj_tensor = torch.arange(-n_ml_half, n_ml_half + odd_mla_shift)
mla_index_tensor = torch.stack(torch.meshgrid(torch.arange(n_micro_lenses), torch.arange(n_micro_lenses)), dim=-1)

# Initialize tensor to store current_offsets
current_offsets_tensor = torch.empty((n_micro_lenses, n_micro_lenses, 2))

# Iterate over each microlens position and compute the offset
for ml_ii_idx, ml_ii in enumerate(ml_ii_tensor):
    for ml_jj_idx, ml_jj in enumerate(ml_jj_tensor):
        current_offset = self._calculate_current_offset(ml_ii.item(), ml_jj.item(), n_voxels_per_ml, n_micro_lenses)
        current_offsets_tensor[ml_ii_idx, ml_jj_idx] = torch.tensor(current_offset)

mla_indices = mla_index_tensor.view(-1, 2)
offsets = current_offsets_tensor.view(-1, 2)

num_lenslets = mla_indices.shape[0]
img_list_all_lenslets = torch.empty(num_lenslets, 2, num_pix, num_pix)
for i in range(num_lenslets):
    ml_jj_idx, ml_ii_idx = mla_indices[i]
    current_offset = offsets[i]
    img_list = self.generate_images(volume_in, current_offset, intensity, mla_index=(ml_jj_idx, ml_ii_idx))
    img_list_all_lenslets[i, ...] = torch.stack(img_list)

lenslets_reshaped = img_list_all_lenslets.view(n_micro_lenses, n_micro_lenses, 2, num_pix, num_pix)

for i in range(n_micro_lenses):
    for j in range(n_micro_lenses):
            start_x = i * num_pix
            start_y = j * num_pix
            end_x = start_x + num_pix
            end_y = start_y + num_pix
            full_img_list_array[:, start_y:end_y, start_x:end_x] = lenslets_reshaped[j, i, :, :, :]
split_img_list_array = torch.split(full_img_list_array, 1, dim=0)
full_img_list = [tensor.squeeze(0) for tensor in split_img_list_array]
        # full_img_list_array[:, j, i] = lenslets_reshaped[j, i, :, :, :]

parallelize_pool = False
if parallelize_pool:
    get_input = self.instance_iterable_inputs_for_image_generation(mla_indices, offsets, volume_in, intensity)
    # with torch.multiprocessing.Pool(processes=1, initializer=init_worker, initargs=(get_input,)) as pool:
    with torch.multiprocessing.Pool(processes=1) as pool:
        results = pool.map(get_input, range(num_lenslets))
    
    
pool_method_attempt = False
if pool_method_attempt:
        with torch.multiprocessing.Pool() as pool:
            process_row = make_process_row_method(self)
            results = [pool.apply(process_row, (ml_ii_idx, row_iterable, n_ml_half, odd_mla_shift, volume_in, intensity)) for ml_ii_idx in range(len(row_iterable))]
            full_img_row_lists = [res.get() for res in results]

        for full_img_row_list in full_img_row_lists:
            full_img_list = self._concatenate_images(full_img_list, full_img_row_list, axis=1)
