import torch
import torch.nn as nn
import torch.nn.functional as f


class CfCCell(nn.Module):

    def __init__(self,
                 input_dim : int = 2,
                 state_dim : int = 2,
                 use_gate : bool = True,
                 backbone_hidden_dim : int = 16,
                 backbone_output_dim : int = 8,
                 backbone_activation_func : nn.Module = nn.SiLU):

        super().__init__()

        self.backbone = nn.Sequential(nn.Linear(in_features=input_dim+state_dim, out_features=backbone_hidden_dim),
                                      backbone_activation_func(),
                                      nn.Linear(in_features=backbone_hidden_dim, out_features=backbone_hidden_dim),
                                      backbone_activation_func(),
                                      nn.Linear(in_features=backbone_hidden_dim, out_features=backbone_output_dim),
                                      nn.Tanh())

        self._use_gate = use_gate
        if not self._use_gate:
            self.alpha = nn.Parameter(data=torch.randn(1, state_dim)*0.1, requires_grad=True)
            self.omega = nn.Parameter(data=torch.zeros(1, state_dim), requires_grad=True)
            self.time = nn.Linear(in_features=backbone_output_dim, out_features=state_dim)
        else:
            self.g = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=state_dim),
                                   nn.Tanh())
            self.h = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=state_dim),
                                   nn.Tanh())

            self.time_as_weight = nn.Linear(in_features=backbone_output_dim, out_features=state_dim)
            self.time_as_bias = nn.Linear(in_features=backbone_output_dim, out_features=state_dim)


    def _forward_gate(self, x, u, dt):
        xu = torch.cat([x, u], dim=-1)
        x_backbone = self.backbone(xu)

        time_w = self.time_as_weight(x_backbone)
        time_b = self.time_as_bias(x_backbone)

        x_f = torch.sigmoid(time_w * dt + time_b)
        x_g = self.g(x_backbone)
        x_h = self.h(x_backbone)

        x_g_prime = x_g * x_f
        x_h_prime = x_h * (1-x_f)
        x_output = x_g_prime + x_h_prime
        return x_output


    def _forward_native(self, x, history, dt):
        xh = torch.cat([x, history], dim=-1)
        x_pos_backbone = self.backbone(xh)
        x_neg_backbone = self.backbone(-xh)

        f_pos = f.softplus(self.time(x_pos_backbone))
        f_neg = torch.sigmoid(self.time(x_neg_backbone))
        omega = f.softplus(self.omega)

        decay = torch.exp( -dt * (omega + f_pos))
        x_output = self.alpha + (history - self.alpha) * f_neg * decay
        return x_output


    def forward(self, x, history, dt):
        """
        x : input at current time step
        u : history
        dt: delta time between current and last observations
        """
        if dt.ndim ==1:
            dt = dt.unsqueeze(-1)
        if self._use_gate is True:
            return self._forward_gate(x, history, dt)
        else:
            return self._forward_native(x, history, dt)


class CfC(nn.Module):

    def __init__(self,
                 input_dim,
                 state_dim,
                 output_dim,
                 use_gate,
                 backbone_hidden_dim,
                 backbone_output_dim,
                 backbone_activation_func,
                 *args, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.cell = CfCCell(input_dim=input_dim,
                            state_dim=state_dim,
                            use_gate=use_gate,
                            backbone_hidden_dim=backbone_hidden_dim,
                            backbone_activation_func=backbone_activation_func,
                            backbone_output_dim=backbone_output_dim)
        self.output = nn.Linear(in_features=state_dim, out_features=output_dim)


    def forward(self, x_sequence, dt_sequence, state=None, return_trajectory=False):
        """
        Notice that dt_sequence should not to be considered here as the timesteps, but the time intervals Δt!
        """
        B, T, D = x_sequence.shape
        if dt_sequence.ndim == 2:
            dt_sequence = dt_sequence.unsqueeze(-1)

        if state is None:
            state = torch.zeros(B, self.state_dim, device=x_sequence.device, dtype=x_sequence.dtype)

        outputs = []
        state_traj = [] if return_trajectory else None
        for t in range(T):
            state = self.cell(x_sequence[:, t], state, dt_sequence[:, t])
            outputs.append(self.output(state))
            if return_trajectory:
                state_traj.append(state)

        out = torch.stack(outputs, dim=1)
        if return_trajectory:
            return out, state, torch.stack(state_traj, dim=1)
        return out, state




if __name__ == "__main__":

    model = CfC(input_dim=3, state_dim=16, output_dim=3, use_gate=False,
                backbone_hidden_dim=32, backbone_output_dim=16,
                backbone_activation_func=nn.SiLU)
    x = torch.randn(4, 20, 3)
    dt = torch.rand(4, 20, 1) * 0.5 + 1e-3
    out, h = model(x, dt)
    out.sum().backward()
    assert out.shape == (4, 20, 3) and h.shape == (4, 16)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in
               model.parameters())
    print(f"Passed all tests!")



