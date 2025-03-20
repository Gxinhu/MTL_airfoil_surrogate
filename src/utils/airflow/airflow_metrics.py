import json
import pyvista as pv
import os
import numpy as np


NU = np.array(1.56e-5)


def compute_metric(output, simulation_name, surf, data_dir, out_coef_norm):
    u_inf = float(simulation_name.split("_")[2])
    angle_of_attack = float(simulation_name.split("_")[3])
    intern: pv.UnstructuredGrid = pv.read(
        os.path.join(
            data_dir,
            simulation_name,
            f"{simulation_name}_internal.vtu",
        )
    )
    aerofoil = pv.read(
        os.path.join(
            data_dir,
            simulation_name,
            f"{simulation_name}_aerofoil.vtp",
        )
    )

    true_coff, true_intern, true_aerofoil = Compute_coefficients(
        intern, aerofoil, surf, u_inf, angle_of_attack, keep_vtk=True
    )

    predict_intern, predict_aerofoil = Airfoil_test(
        intern, aerofoil, output, out_coef_norm, surf
    )

    (
        predict_coff,
        predict_intern,
        predict_aerofoil,
    ) = Compute_coefficients(
        predict_intern,
        predict_aerofoil,
        surf,
        u_inf,
        angle_of_attack,
        keep_vtk=True,
    )

    rel_err_force = np.abs((true_coff - predict_coff) / true_coff)
    rel_err_wss = np.abs(
        (
            true_aerofoil.point_data["wallShearStress"]
            - predict_aerofoil.point_data["wallShearStress"]
        )
        / true_aerofoil.point_data["wallShearStress"]
    ).mean(axis=0)
    rel_err_p = np.abs(
        (true_aerofoil.point_data["p"] - predict_aerofoil.point_data["p"])
        / true_aerofoil.point_data["p"]
    ).mean(axis=0)
    return true_coff, predict_coff, rel_err_force, rel_err_wss, rel_err_p


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rsquared(predict, true):
    """
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    """
    mean = true.mean(dim=0)
    return 1 - ((true - predict) ** 2).sum(dim=0) / ((true - mean) ** 2).sum(
        dim=0
    )


def rel_err(a, b):
    return np.abs((a - b) / a)


def WallShearStress(Jacob_U, normals):
    S = 0.5 * (Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    ShearStress = 2 * NU.reshape(-1, 1, 1) * S
    ShearStress = (ShearStress * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)

    return ShearStress


def Airfoil_test(internal, airfoil, out, coef_norm, bool_surf):
    # Produce multiple copies of a simulation for different predictions.
    # stocker les internals, airfoils, calculer le wss, calculer le drag, le lift, plot pressure coef, plot skin friction coef, plot drag/drag, plot lift/lift
    # calcul spearsman coef, boundary layer
    intern_copy = internal.copy()
    aerofoil_copy = airfoil.copy()

    point_mesh = intern_copy.points[bool_surf, :2]
    point_surf = aerofoil_copy.points[:, :2]
    out = (out * (coef_norm[1] + 1e-8) + coef_norm[0]).cpu().numpy()
    out[bool_surf, :2] = np.zeros_like(out[bool_surf, :2])
    out[bool_surf, 3] = np.zeros_like(out[bool_surf, 3])
    intern_copy.point_data["U"][:, :2] = out[:, :2]
    intern_copy.point_data["p"] = out[:, 2]
    intern_copy.point_data["nut"] = out[:, 3]

    surf_p = intern_copy.point_data["p"][bool_surf]
    surf_p = reorganize(point_mesh, point_surf, surf_p)
    aerofoil_copy.point_data["p"] = surf_p
    intern_copy = intern_copy.ptc(pass_point_data=True)
    aerofoil_copy = aerofoil_copy.ptc(pass_point_data=True)

    return intern_copy, aerofoil_copy


def Compute_coefficients(
    internal, airfoil, bool_surf, u_inf, angle, keep_vtk=False
):
    interal_copy = internal.copy()
    aerofoil_copy = airfoil.copy()
    point_mesh = interal_copy.points[bool_surf, :2]
    point_surf = aerofoil_copy.points[:, :2]

    interal_copy = interal_copy.compute_derivative(
        scalars="U", gradient="pred_grad"
    )

    surf_grad = interal_copy.point_data["pred_grad"].reshape(-1, 3, 3)[
        bool_surf, :2, :2
    ]
    surf_p = interal_copy.point_data["p"][bool_surf]
    surf_grad = reorganize(point_mesh, point_surf, surf_grad)
    surf_p = reorganize(point_mesh, point_surf, surf_p)

    Wss_pred = WallShearStress(surf_grad, -aerofoil_copy.point_data["Normals"])
    aerofoil_copy.point_data["wallShearStress"] = Wss_pred
    aerofoil_copy.point_data["p"] = surf_p

    interal_copy = interal_copy.ptc(pass_point_data=True)
    aerofoil_copy = aerofoil_copy.ptc(pass_point_data=True)

    WP_int = (
        -aerofoil_copy.cell_data["p"][:, None]
        * aerofoil_copy.cell_data["Normals"][:, :2]
    )

    Wss_int = (
        aerofoil_copy.cell_data["wallShearStress"]
        * aerofoil_copy.cell_data["Length"].reshape(-1, 1)
    ).sum(axis=0)
    WP_int = (WP_int * aerofoil_copy.cell_data["Length"].reshape(-1, 1)).sum(
        axis=0
    )
    force = Wss_int - WP_int

    alpha = angle * np.pi / 180
    basis = np.array(
        [[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]
    )
    force_rot = basis @ force
    coef = 2 * force_rot / u_inf**2
    return (coef, interal_copy, aerofoil_copy) if keep_vtk else coef


def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    point_to_idx = {tuple(in_order_points[i]): i for i in range(n)}
    for i in range(n):
        idx[i] = point_to_idx[tuple(out_order_points[i])]
    idx = idx.astype(int)
    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reordered[idx]
