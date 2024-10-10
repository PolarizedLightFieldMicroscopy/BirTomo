import numpy as np


def ret_and_azim_from_intensity(image_list, swing):
    """Note: this function is still in development."""
    if len(image_list) != 5:
        raise ValueError(f"Expected 5 images, got {len(image_list)}.")
    # The order of the images matters!
    imgs = [image_list[0], image_list[2], image_list[3], image_list[1], image_list[4]]
    # using arctan vs arctan2 does not seem to make a difference
    epsilon = np.finfo(float).eps
    # A = (imgs[1] - imgs[2]) / (imgs[1] + imgs[2] - 2 * imgs[0] + epsilon) * np.arctan(swing / 2)
    # B = (imgs[4] - imgs[3]) / (imgs[4] + imgs[3] - 2 * imgs[0] + epsilon) * np.arctan(swing / 2)
    # Ensure that the denominator is not zero by adding epsilon
    denominator_A = imgs[1] + imgs[2] - 2 * imgs[0] + epsilon
    denominator_B = imgs[4] + imgs[3] - 2 * imgs[0] + epsilon
    # Check where the denominator is zero and set those values to epsilon
    denominator_A[denominator_A == 0] = epsilon
    denominator_B[denominator_B == 0] = epsilon
    # Calculate A and B with the safe denominator
    A = (imgs[1] - imgs[2]) / denominator_A * np.arctan(swing / 2)
    B = (imgs[4] - imgs[3]) / denominator_B * np.arctan(swing / 2)
    # ret = np.arctan(np.sqrt(A ** 2, B ** 2))
    ret = np.arctan(np.sqrt(A**2 + B**2))
    test_value = imgs[1] + imgs[2] - 2 * imgs[0]
    indices = np.where(test_value < 0)
    ret[indices] = 2 * np.pi - ret[indices]
    # azim = 0.5 * np.atan2(A, B) + np.pi / 2
    azim = 0.5 * np.atan2(B, A) + np.pi / 2
    return [ret, azim]
