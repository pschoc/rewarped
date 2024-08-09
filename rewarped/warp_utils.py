import warp as wp


@wp.kernel
def eval_kinematic(
    joint_q: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    p: float,
    # outputs
    joint_q_next: wp.array(dtype=float),
):
    tid = wp.tid()
    joint_q_next[tid] = joint_q[tid] + p * joint_act[tid]  # relative


def eval_kinematic_fk(model, state_in, state_out, sim_dt, sim_substeps, control):
    wp.launch(
        kernel=eval_kinematic,
        dim=model.joint_axis_count,
        inputs=[state_in.joint_q, control.joint_act, float(1.0 / sim_substeps)],
        outputs=[state_out.joint_q],
        device=model.device,
    )
    wp.sim.eval_fk(model, state_out.joint_q, state_out.joint_qd, None, state_out)
