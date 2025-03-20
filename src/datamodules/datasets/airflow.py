import numpy as np
import pyvista as pv
import os

import torch
from src.utils.airflow.airflow_metrics import reorganize
from tqdm.rich import tqdm


class Dataset:
    def __init__(self, pos, x, y, surf):
        self.pos = pos
        self.x = x
        self.y = y
        self.surf = surf


def AirflowDatasetBuilder(
    sets,
    crop=None,
    base_folder="",
    save_folder="",
):
    """
    Create a list of simulation to input in a PyTorch Geometric DataLoader.
    Simulation are transformed by keeping vertices of the CFD mesh or by sampling
    (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``,
            the mean and the standard deviation of the dataset will be computed and returned.
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None.
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    """

    for _, s in enumerate(tqdm(sets)):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read(f"{base_folder}/{s}/{s}_internal.vtu")
        aerofoil = pv.read(f"{base_folder}/{s}/{s}_aerofoil.vtp")
        internal = internal.compute_cell_sizes(length=False, volume=False)

        # Cropping if needed, crinkle is True.
        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(
                bounds=bounds, invert=False, crinkle=True
            )
        surf_bool = internal.point_data["U"][:, 0] == 0
        geom = -internal.point_data["implicit_distance"][
            :, None
        ]  # Signed distance function
        u_inf, alpha = (
            float(s.split("_")[2]),
            float(s.split("_")[3]) * np.pi / 180,
        )
        u = (np.array([np.cos(alpha), np.sin(alpha)]) * u_inf).reshape(
            1, 2
        ) * np.ones_like(internal.point_data["U"][:, :1])
        normal = np.zeros_like(u)
        normal[surf_bool] = reorganize(
            aerofoil.points[:, :2],
            internal.points[surf_bool, :2],
            -aerofoil.point_data["Normals"][:, :2],
        )

        attr = np.concatenate(
            [
                u,
                geom,
                normal,
                internal.point_data["U"][:, :2],
                internal.point_data["p"][:, None],
                internal.point_data["nut"][:, None],
            ],
            axis=-1,
        )

        pos = internal.points[:, :2]
        init = np.concatenate([pos, attr[:, :5]], axis=1)
        target = attr[:, 5:]

        # Put everything in tensor
        surf = torch.tensor(surf_bool)
        pos = torch.tensor(pos, dtype=torch.float)
        x = torch.tensor(init, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)

        # Graph definition
        # if cell_centers:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool(), centers = centers.bool())
        # else:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool())

        datas = Dataset(pos=pos, x=x, y=y, surf=surf.bool())
        torch.save(datas, os.path.join(save_folder, f"{s}.pth"))
