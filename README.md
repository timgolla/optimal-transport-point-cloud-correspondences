# Optimal transport-based correspondence computation for labeled point clouds

This is research code I wrote based on the work presented in my paper

Golla, T., Kneiphof, T., Kuhlmann, H., Weinmann, M. and Klein, R. (2020), *Temporal Upsampling of Point Cloud Sequences by Optimal Transport for Plant Growth Visualization*. Computer Graphics Forum, 39: 167-179. https://doi.org/10.1111/cgf.14009

This code accepts two point cloud files and two label files as input and computes non-injective correspondces for them, using optimal transport. Execute `python ot-correspondences.py` and supply the point cloud files and label files as input, as well as the output file name.