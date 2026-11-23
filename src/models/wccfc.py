import torch
import torch.nn as nn
import torch.nn.functional as f


class WC_CfCCell(nn.Module):

    def __init__(self,
                 input_dim : int = 2,
                 state_dim : int = 2,
                 use_gate : bool = True,
                 backbone_hidden_dim : int = 16,
                 backbone_output_dim : int = 8,
                 backbone_activation_func : nn.Module = nn.SiLU):

        super().__init__()

        self._use_gate = use_gate

        init_val = torch.tensor(1.0 / (state_dim ** 0.5))
        self.w_ee = nn.Parameter(init_val + 0.1 * torch.randn(state_dim, state_dim))
        self.w_ei = nn.Parameter(init_val + 0.1 * torch.randn(state_dim, state_dim))
        self.w_ie = nn.Parameter(init_val + 0.1 * torch.randn(state_dim, state_dim))
        self.w_ii = nn.Parameter(init_val + 0.1 * torch.randn(state_dim, state_dim))
        self.w_eu = nn.Linear(input_dim, state_dim, bias=True)
        self.w_iu = nn.Linear(input_dim, state_dim, bias=True)

        self.backbone = nn.Sequential(nn.Linear(in_features=2*state_dim, out_features=backbone_hidden_dim),
                                      backbone_activation_func(),
                                      nn.Linear(in_features=backbone_hidden_dim, out_features=backbone_output_dim),
                                      nn.Tanh())

        self.g = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=2*state_dim), nn.Tanh())
        self.h = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=2*state_dim), nn.Tanh())

        self.omega = nn.Linear(in_features=backbone_output_dim, out_features=state_dim)

        if not self._use_gate:
            self.gamma = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=state_dim), nn.Softplus())
            nn.init.constant_(self.gamma[0].bias, -2.0)
        else:
            self.gamma_as_weight = nn.Sequential(nn.Linear(in_features=backbone_output_dim, out_features=state_dim), nn.Softplus())
            self.gamma_as_bias = nn.Linear(in_features=backbone_output_dim, out_features=state_dim)


    def forward(self, x, state, dt):
        """
        x : input at current time step
        u : history
        dt: delta time between current and last observations
        """
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)

        E, I = state
        s_E = f.linear(E, f.softplus(self.w_ee)) - f.linear(I, f.softplus(self.w_ei)) + self.w_eu(x)
        s_I = f.linear(E, f.softplus(self.w_ie)) - f.linear(I, f.softplus(self.w_ii)) + self.w_iu(x)
        s = torch.cat([s_E, s_I], dim=-1)

        x_backbone = self.backbone(s)

        if self._use_gate is False:
            gamma = self.gamma(x_backbone)
            d = torch.exp(- gamma * dt)
        else:
            gamma_w = self.gamma_as_weight(x_backbone)
            gamma_b = self.gamma_as_bias(x_backbone)
            d = torch.sigmoid(-gamma_w * dt + gamma_b)

        omega = f.softplus(self.omega(x_backbone))
        cos, sin = d * torch.cos(omega * dt), d * torch.sin(omega * dt)

        h_E, h_I = self.h(x_backbone).chunk(2, dim=-1)
        g_E, g_I = self.g(x_backbone).chunk(2, dim=-1)

        dE, dI = g_E-h_E, g_I-h_I
        E_next = h_E + cos * dE - sin * dI
        I_next = h_I + sin * dE + cos * dI

        return E_next, I_next



class WC_CfC(nn.Module):

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
        self.cell = WC_CfCCell(input_dim=input_dim,
                               state_dim=state_dim,
                               use_gate=use_gate,
                               backbone_hidden_dim=backbone_hidden_dim,
                               backbone_output_dim=backbone_output_dim,
                               backbone_activation_func=backbone_activation_func)
        self.output = nn.Linear(in_features=2*state_dim, out_features=output_dim)


    def forward(self, x_sequence, dt_sequence):
        """
        Notice that dt_sequence should not to be considered here as the timesteps, but the time intervals Δt!
        """
        B, T, D = x_sequence.shape
        if dt_sequence.ndim == 2:
            dt_sequence = dt_sequence.unsqueeze(-1)

        # Initialize empty history
        states_E = torch.zeros(B, self.state_dim, device=x_sequence.device, dtype=x_sequence.dtype)
        states_I = torch.zeros(B, self.state_dim, device=x_sequence.device, dtype=x_sequence.dtype)

        # Initialize empty outputs
        outputs = []
        # Recurrent step
        for t in range(T):
            states_E, states_I = self.cell(x_sequence[:, t], (states_E, states_I), dt_sequence[:, t])
            print(f"E: {states_E}")
            print(f"I: {states_I}")
            outputs.append(self.output(torch.cat([states_E, states_I], dim=-1)))

        history = [states_E, states_I]
        return torch.stack(outputs, dim=1), history


if __name__ == "__main__":

    model = WC_CfC(input_dim=1, state_dim=8, output_dim=2, use_gate=False,
                backbone_hidden_dim=32, backbone_output_dim=16,
                backbone_activation_func=nn.SiLU)
    x = torch.randn(4, 20, 1)
    dt = torch.rand(4, 20, 1) * 0.5 + 1e-3
    out, h = model(x, dt)
    out.sum().backward()
    assert out.shape == (4, 20, 2), f"Expected (4, 20, 2), got : {out.shape}"
    assert h[0].shape == (4, 8) and h[1].shape == (4, 8), f"Expected (4, 16) for both, got {h[0].shape},{h[1].shape}"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in
               model.parameters())
    print(f"Passed all tests!")