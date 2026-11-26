"""Joint refinement of the arms, fingers and object scale."""
import torch
from tqdm import tqdm, trange

from optimizer.handoptim import HandOptimizer
from optimizer.twohandoptim import TwoHandOptimizer

JOINT_PATIENCE = 100    # early stop patience
JOINT_MIN_STEPS = 250   # no early stop before this step
PEN_DONE = 1e-6         # no early stop above this penetration loss (m^2)


def optimize_joint(human, smplh_params, smplh_verts, obj_verts, obj_faces,
                   left_contact, right_contact, left_kpts, right_kpts,
                   contact_pairs, out_cam, batch, output_cam_trans, focal_length_, amodal_mask):
    """Refine contacting arms, fingers and object scale."""
    left_hopt = left_contact and (left_kpts is not None)
    right_hopt = right_contact and (right_kpts is not None)
    if not (left_hopt or right_hopt):
        return smplh_params, smplh_verts, obj_verts

    if left_hopt and right_hopt:
        two_hand_config = {
            'focal_length': focal_length_,
            'obj_faces': obj_faces,
            'obj_verts': obj_verts,
            'smplh_params': smplh_params,
            'smplh_verts': smplh_verts,
            'left_kpts': left_kpts,
            'right_kpts': right_kpts,
            'cam': out_cam,
            'batch': batch,
            'amodal_mask': amodal_mask,
            'contact_pairs': contact_pairs,
        }
        print("[joint] both hands in contact: coarse two-arm stage", flush=True)
        optimizer_model = TwoHandOptimizer(**two_hand_config)
        optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=1e-2)

        best_loss = float('inf')
        best_state_dict = None
        named_params = dict(optimizer_model.named_parameters())
        since_best = 0
        best = None
        for step in trange(500, desc="Two-arm stage"):
            optimizer.zero_grad()

            kp_loss, pen_loss, tact_loss, reg_loss = optimizer_model.all_losses()
            loss = kp_loss + 5e5 * pen_loss + 1e4 * tact_loss + 100 * reg_loss

            current = loss.item()
            if current < best_loss:                 # snapshot before the step
                best_state_dict = {k: v.clone().detach() for k, v in named_params.items()}
            loss.backward()
            optimizer.step()

            if current < best_loss:
                best_loss = current
                since_best = 0
                best = (step, kp_loss.item(), pen_loss.item(), tact_loss.item(), optimizer_model.n_tunnel)
            else:
                since_best += 1
                if step >= JOINT_MIN_STEPS and since_best >= JOINT_PATIENCE and not (best[2] > PEN_DONE):
                    tqdm.write(f"[joint] two-arm stage: early stop at step {step} "
                               f"(no improvement for {since_best})")
                    break

        optimizer_model.load_state_dict(best_state_dict, strict=False)   # parameters only
        print(f"[joint] two-arm stage: best state at step {best[0] + 1}: kp={best[1]:.4f} pen={best[2]:.3e} "
              f"contact={best[3]:.3e} tunneled fingers={best[4]}", flush=True)
        smplh_params_two = optimizer_model()
        smplh_params = {k: v.detach().clone() for k, v in smplh_params_two.items()}
        output_vertices, _, _ = human.run_body_model(human.smplh_model, smplh_params, out_cam, batch)
        smplh_verts = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()

    hand_config = {
        'focal_length': focal_length_,
        'obj_faces': obj_faces,
        'obj_verts': obj_verts,
        'smplh_params': smplh_params,
        'smplh_verts': smplh_verts,
        'left_kpts': left_kpts,
        'right_kpts': right_kpts,
        'cam': out_cam,
        'batch': batch,
        'left_flag': left_hopt,
        'right_flag': right_hopt,
        'amodal_mask': amodal_mask,
        'contact_pairs': contact_pairs,
    }
    sides = [s for s, h in (("left", left_hopt), ("right", right_hopt)) if h]
    print(f"[joint] standard stage: {' + '.join(sides)} hand", flush=True)
    optimizer_model = HandOptimizer(**hand_config)
    optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=3e-3)

    best_loss = float('inf')
    best_state_dict = None
    named_params = dict(optimizer_model.named_parameters())
    since_best = 0
    best = None
    for step in trange(500, desc="Joint refinement"):
        optimizer.zero_grad()

        kp_loss, pen_loss, reg_loss, tact_loss = optimizer_model.all_losses()
        loss = 0.01 * kp_loss + 1.5e5 * pen_loss + 100 * reg_loss + 3e4 * tact_loss

        current = loss.item()
        if current < best_loss:                     # snapshot before the step
            best_state_dict = {k: v.clone().detach() for k, v in named_params.items()}
        loss.backward()
        optimizer.step()
        # clamp the object scale
        with torch.no_grad():
            optimizer_model.move_coeff.data.clamp_(0.95, 1.05)

        if current < best_loss:
            best_loss = current
            since_best = 0
            best = (step, kp_loss.item(), pen_loss.item(), tact_loss.item(), optimizer_model.n_tunnel)
        else:
            since_best += 1
            if step >= JOINT_MIN_STEPS and since_best >= JOINT_PATIENCE and not (best[2] > PEN_DONE):
                tqdm.write(f"[joint] standard stage: early stop at step {step} "
                           f"(no improvement for {since_best})")
                break

    optimizer_model.load_state_dict(best_state_dict, strict=False)   # parameters only
    print(f"[joint] standard stage: best state at step {best[0] + 1}: kp={best[1]:.4f} pen={best[2]:.3e} "
          f"contact={best[3]:.3e} tunneled fingers={best[4]}", flush=True)
    smplh_params, move_coeff = optimizer_model()
    output_vertices, _, _ = human.run_body_model(human.smplh_model, smplh_params, out_cam, batch)
    smplh_verts = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
    s_o = move_coeff.detach().cpu().item()
    print(f"[joint] object scale s_o = {s_o:.4f}", flush=True)
    obj_verts = s_o * obj_verts
    smplh_params = {k: v.detach().clone() for k, v in smplh_params.items()}
    return smplh_params, smplh_verts, obj_verts
