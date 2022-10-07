from warm_pixels.hst_functions.trail_model import trail_model


class TrailModel:
    def __init__(
            self,
            rho_q,
            beta,
            w,
            a,
            b,
            c,
            tau_a,
            tau_b,
            tau_c,
    ):
        self.rho_q = rho_q
        self.beta = beta
        self.w = w
        self.a = a
        self.b = b
        self.c = c
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.tau_c = tau_c

    def __call__(self, x, n_e, n_bg, row):
        return trail_model(
            x=x,
            rho_q=self.rho_q,
            n_e=n_e,
            n_bg=n_bg,
            row=row,
            beta=self.beta,
            w=self.w,
            A=self.a,
            B=self.b,
            C=self.c,
            tau_a=self.tau_a,
            tau_b=self.tau_b,
            tau_c=self.tau_c,
        )