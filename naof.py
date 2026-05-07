import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_in = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm_hidden = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm_hidden(x)
        x = self.fc2(x)
        return x


class Combiner(nn.Module):



    # =========================
    NUM_EXPERTS = 8

    #  BEiT3-Base: d = 768
    # router: 2d -> d -> K
    ROUTER_HIDDEN_MULT = 1.0

    # expert: 2d -> d -> d
    EXPERT_HIDDEN_MULT = 4.0

    # Prevent the hidden dimension from becoming too small when using a smaller backbone.
    MIN_HIDDEN_DIM = 256

    DROPOUT = 0.2
    EPS = 1e-6

    # Set to -2.0 so the initial gate is approximately sigmoid(-2)=0.119.
    # This biases early training toward the text anchor, then gradually learns to add the proxy residual.
    # Set this to 0.0 if you prefer an initial gate close to 0.5.
    GATE_BIAS_INIT = -2.0

    def __init__(self, feature_dim: int, *legacy_unused_dims):
        """
        Args:
            feature_dim:
                BEiT3 retrieval feature dim.
                This is 768 for BEiT3-Base and 1024 for BEiT3-Large.

            *legacy_unused_dims:
                Kept for compatibility with existing code:
                    Combiner(feature_dim, projection_dim, hidden_dim)

                This accepts projection_dim / hidden_dim but does not use them.
                Therefore, projection_dim and hidden_dim in Config no longer control the Combiner.
        """
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.feature_dim = int(feature_dim)
        self.input_dim = self.feature_dim * 2
        self.num_experts = self.NUM_EXPERTS
        self.eps = self.EPS

        # Automatically determine the internal hidden dimensions.
        self.router_hidden_dim = self._make_hidden_dim(self.ROUTER_HIDDEN_MULT)
        self.expert_hidden_dim = self._make_hidden_dim(self.EXPERT_HIDDEN_MULT)

        # R_theta(u): R^{2d} -> R^K
        self.router = TwoLayerMLP(
            input_dim=self.input_dim,
            hidden_dim=self.router_hidden_dim,
            output_dim=self.num_experts,
            dropout=self.DROPOUT,
        )

        # E_k(u): R^{2d} -> R^d
        # The K experts share the same architecture but use independent parameters.
        self.experts = nn.ModuleList([
            TwoLayerMLP(
                input_dim=self.input_dim,
                hidden_dim=self.expert_hidden_dim,
                output_dim=self.feature_dim,
                dropout=self.DROPOUT,
            )
            for _ in range(self.num_experts)
        ])

        self.reset_parameters()

    def _make_hidden_dim(self, multiplier: float) -> int:
        hidden_dim = int(round(self.feature_dim * multiplier))
        hidden_dim = max(hidden_dim, self.MIN_HIDDEN_DIM)
        return hidden_dim

    def reset_parameters(self):
        """
        Initialization strategy:
        - Initialize Linear weights with trunc_normal.
        - Initialize Linear biases to 0 by default.
        - LayerNorm weight=1, bias=0
        - Set the final-layer bias of each expert to GATE_BIAS_INIT to make the initial gate conservative.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        # Only adjust the final-layer bias of each expert.
        # This prevents the initial gate from overusing the generated proxy residual.
        for expert in self.experts:
            if expert.fc2.bias is not None:
                nn.init.constant_(expert.fc2.bias, self.GATE_BIAS_INIT)

    def _check_inputs(
        self,
        proxy_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        if proxy_features is None:
            raise ValueError("proxy_features must not be None.")
        if text_features is None:
            raise ValueError("text_features must not be None.")

        if proxy_features.dim() != 2:
            raise ValueError(
                f"proxy_features must have shape [B, d], got {tuple(proxy_features.shape)}"
            )

        if text_features.dim() != 2:
            raise ValueError(
                f"text_features must have shape [B, d], got {tuple(text_features.shape)}"
            )

        if proxy_features.shape != text_features.shape:
            raise ValueError(
                "proxy_features and text_features must have the same shape. "
                f"Got proxy_features={tuple(proxy_features.shape)}, "
                f"text_features={tuple(text_features.shape)}."
            )

        if proxy_features.size(-1) != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.feature_dim}, "
                f"got {proxy_features.size(-1)}."
            )

    def forward(
        self,
        proxy_features: torch.Tensor,
        text_features: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args:
            proxy_features:
                shape [B, d].
                Corresponds to reference_features in the training code.
                Corresponds to gen_features in the validation code.

            text_features:
                shape [B, d].

            return_aux:
                Whether to return intermediate variables for debugging gates and routing weights.

        Returns:
            fused_features:
                shape [B, d].
        """
        self._check_inputs(proxy_features, text_features)

        # ------------------------------------------------------------
        # 1. Normalize
        # ------------------------------------------------------------
        # z_P: [B, d]
        # z_T: [B, d]
        z_proxy = F.normalize(proxy_features, dim=-1, eps=self.eps)
        z_text = F.normalize(text_features, dim=-1, eps=self.eps)

        # ------------------------------------------------------------
        # 2. Orthogonal proxy residual
        # ------------------------------------------------------------
        # dot: [B, 1]
        # Compute (z_T)^T z_P separately for each sample.
        dot = torch.sum(z_text * z_proxy, dim=-1, keepdim=True)

        # d_P: [B, d]
        # Remove the component of the proxy aligned with text and keep only complementary information outside the text direction.
        proxy_residual = z_proxy - dot * z_text

        # ------------------------------------------------------------
        # 3. Joint representation u = [z_T ; d_P]
        # ------------------------------------------------------------
        # joint: [B, 2d]
        # BEiT3-Base: [B, 1536]
        joint = torch.cat([z_text, proxy_residual], dim=-1)

        # ------------------------------------------------------------
        # 4. Routing network
        # ------------------------------------------------------------
        # routing_logits: [B, K]
        # BEiT3-Base + K=8: [B, 8]
        routing_logits = self.router(joint)

        # routing_weights pi: [B, K]
        routing_weights = F.softmax(routing_logits, dim=-1)

        # ------------------------------------------------------------
        # 5. Expert networks
        # ------------------------------------------------------------
        # Each expert outputs [B, d].
        # After stacking, expert_logits has shape [B, K, d].
        expert_logits = torch.stack(
            [expert(joint) for expert in self.experts],
            dim=1,
        )

        # ------------------------------------------------------------
        # 6. Mixture-of-gates
        # ------------------------------------------------------------
        # routing_weights.unsqueeze(-1): [B, K, 1]
        # mixed_gate_logits: [B, d]
        mixed_gate_logits = torch.sum(
            routing_weights.unsqueeze(-1) * expert_logits,
            dim=1,
        )

        # gate a: [B, d], one reliability weight for each feature dimension.
        gate = torch.sigmoid(mixed_gate_logits)

        # ------------------------------------------------------------
        # 7. Final fused query
        # ------------------------------------------------------------
        # z_F = normalize(z_T + a * d_P)
        # fused_features: [B, d]
        fused_features = F.normalize(
            z_text + gate * proxy_residual,
            dim=-1,
            eps=self.eps,
        )

        if not return_aux:
            return fused_features

        aux = {
            "z_text": z_text.detach(),
            "z_proxy": z_proxy.detach(),
            "dot_text_proxy": dot.detach(),
            "proxy_residual": proxy_residual.detach(),
            "joint": joint.detach(),
            "routing_logits": routing_logits.detach(),
            "routing_weights": routing_weights.detach(),
            "expert_logits": expert_logits.detach(),
            "mixed_gate_logits": mixed_gate_logits.detach(),
            "gate": gate.detach(),
            "gate_mean": gate.detach().mean(),
            "gate_min": gate.detach().min(),
            "gate_max": gate.detach().max(),
        }

        return fused_features, aux

    def combine_features(
        self,
        proxy_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interface kept compatible with the existing training/validation code.

        The call pattern in this project is:
            combiner.combine_features(reference_features, text_features)
            combiner.combine_features(gen_features, text_features)

        Therefore, the first argument is the generated visual proxy / reference image feature,
        and the second argument is the text feature.
        """
        return self.forward(
            proxy_features=proxy_features,
            text_features=text_features,
            return_aux=False,
        )

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, "
            f"input_dim={self.input_dim}, "
            f"num_experts={self.num_experts}, "
            f"router_hidden_dim={self.router_hidden_dim}, "
            f"expert_hidden_dim={self.expert_hidden_dim}, "
            f"dropout={self.DROPOUT}, "
            f"gate_bias_init={self.GATE_BIAS_INIT}"
        )