"""
===============================================================================
SOFT ROBOT VLA
===============================================================================

Compact hierarchical Vision-Language-Action model for soft-robot trajectory
tracking.

Primary output:
    A short chunk of future Cartesian waypoint OFFSETS.

Secondary output:
    Pressure delta auxiliary prediction.

The intended control hierarchy is:

    image
    + instruction
    + robot state
    + trajectory context
            |
            v
          VLA
            |
            v
    future Cartesian waypoints
            |
            v
       inverse MLP
            |
            v
       pressure P1/P2/P3
            |
            v
        soft robot

Design principles inspired by:

    RT-1
    OpenVLA
    OpenVLA-OFT
    ACT
    BAKU
    Diffusion Policy
    learned soft-robot trajectory controllers

This is NOT a 7B foundation VLA. It is a task-specific compact VLA
designed for the available soft-robot dataset and hardware.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# =============================================================================
# DEFAULT DIMENSIONS
# =============================================================================

IMAGE_SIZE = 224

HISTORY_LENGTH = 4

ACTION_CHUNK = 8

MAX_INSTRUCTION_LENGTH = 24

STATE_DIM = 9

TRAJECTORY_DIM = 14

ACTION_DIM = 3


# =============================================================================
# ACTIVATION
# =============================================================================

def make_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    dropout=0.05,
):

    return nn.Sequential(

        nn.Linear(
            input_dim,
            hidden_dim,
        ),

        nn.LayerNorm(
            hidden_dim,
        ),

        nn.GELU(),

        nn.Dropout(
            dropout,
        ),

        nn.Linear(
            hidden_dim,
            output_dim,
        ),

        nn.LayerNorm(
            output_dim,
        ),

        nn.GELU(),
    )


# =============================================================================
# VISUAL ENCODER
# =============================================================================

class VisualEncoder(nn.Module):
    """
    ResNet18 visual encoder.

    The final classification layer is removed.

    Output:
        [B, visual_dim]
    """

    def __init__(
        self,
        output_dim=256,
        pretrained=False,
        freeze_backbone=False,
    ):

        super().__init__()

        if pretrained:

            try:

                weights = (
                    ResNet18_Weights.IMAGENET1K_V1
                )

                backbone = resnet18(
                    weights=weights
                )

            except Exception:

                print(
                    "WARNING: Could not load pretrained "
                    "ResNet18 weights."
                )

                print(
                    "Falling back to randomly initialized "
                    "ResNet18."
                )

                backbone = resnet18(
                    weights=None
                )

        else:

            backbone = resnet18(
                weights=None
            )

        self.backbone = nn.Sequential(
            *list(
                backbone.children()
            )[:-1]
        )

        self.projection = nn.Sequential(

            nn.Linear(
                512,
                output_dim,
            ),

            nn.LayerNorm(
                output_dim,
            ),

            nn.GELU(),
        )

        if freeze_backbone:

            for parameter in (
                self.backbone.parameters()
            ):

                parameter.requires_grad = False

    def forward(
        self,
        image,
    ):

        features = self.backbone(
            image
        )

        features = features.flatten(
            start_dim=1
        )

        return self.projection(
            features
        )


# =============================================================================
# LANGUAGE ENCODER
# =============================================================================

class LanguageEncoder(nn.Module):
    """
    Small task-specific language encoder.

    We intentionally do not use a 7B language model.

    For this project the language vocabulary consists primarily of
    trajectory/task descriptions such as:

        follow a circle
        follow a figure eight
        track a heart trajectory
        follow a complex curved path

    A small Transformer is therefore sufficient.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=192,
        output_dim=256,
        num_heads=4,
        num_layers=2,
        max_length=MAX_INSTRUCTION_LENGTH,
        dropout=0.05,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.position_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    max_length,
                    embedding_dim,
                )
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.output_projection = nn.Sequential(

            nn.Linear(
                embedding_dim,
                output_dim,
            ),

            nn.LayerNorm(
                output_dim,
            ),

            nn.GELU(),
        )

    def forward(
        self,
        tokens,
    ):

        x = self.embedding(
            tokens
        )

        length = x.shape[1]

        x = (
            x
            +
            self.position_embedding[
                :,
                :length,
                :
            ]
        )

        padding_mask = (
            tokens == 0
        )

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        valid = (
            ~padding_mask
        ).float().unsqueeze(-1)

        denominator = (
            valid.sum(
                dim=1
            ).clamp(
                min=1.0
            )
        )

        x = (
            x * valid
        ).sum(
            dim=1
        ) / denominator

        return self.output_projection(
            x
        )


# =============================================================================
# PROPRIOCEPTION ENCODER
# =============================================================================

class ProprioceptionEncoder(nn.Module):
    """
    Encodes:

        position       3
        velocity       3
        pressure       3

    Total:
        9 dimensions
    """

    def __init__(
        self,
        input_dim=STATE_DIM,
        output_dim=256,
    ):

        super().__init__()

        self.network = make_mlp(
            input_dim,
            256,
            output_dim,
        )

    def forward(
        self,
        state,
    ):

        return self.network(
            state
        )


# =============================================================================
# TRAJECTORY ENCODER
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Trajectory context:

        target position          3
        next target              3
        target error             3
        tangent                  3
        curvature                1
        speed                    1
        progress                 1

    Total:
        15 dimensions

    The model accepts TRAJECTORY_DIM=14 by default because the current
    dataset interface may omit one scalar. The train script constructs
    exactly the configured representation.
    """

    def __init__(
        self,
        input_dim=TRAJECTORY_DIM,
        output_dim=256,
    ):

        super().__init__()

        self.network = make_mlp(
            input_dim,
            256,
            output_dim,
        )

    def forward(
        self,
        trajectory,
    ):

        return self.network(
            trajectory
        )


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        dim,
        dropout=0.05,
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.fc1 = nn.Linear(
            dim,
            dim * 2,
        )

        self.fc2 = nn.Linear(
            dim * 2,
            dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
    ):

        residual = x

        x = self.norm(
            x
        )

        x = self.fc1(
            x
        )

        x = F.gelu(
            x
        )

        x = self.dropout(
            x
        )

        x = self.fc2(
            x
        )

        return (
            residual
            +
            self.dropout(x)
        )


# =============================================================================
# TEMPORAL FUSION
# =============================================================================

class TemporalObservationEncoder(nn.Module):
    """
    Encodes a sequence of observation tokens.

    Each timestep receives:

        visual
        proprioception
        trajectory

    These are combined into one timestep token and passed through a
    Transformer encoder.

    This is important for soft robots because position alone is not enough:
    velocity, pressure history and trajectory evolution contain information
    about dynamic lag and hysteresis.
    """

    def __init__(
        self,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        history_length=HISTORY_LENGTH,
        dropout=0.05,
    ):

        super().__init__()

        self.input_projection = nn.Sequential(

            nn.Linear(
                256 + 256 + 256,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.GELU(),
        )

        self.time_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    history_length,
                    hidden_dim,
                )
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

    def forward(
        self,
        visual,
        proprio,
        trajectory,
    ):

        x = torch.cat(
            [
                visual,
                proprio,
                trajectory,
            ],
            dim=-1,
        )

        x = self.input_projection(
            x
        )

        length = x.shape[1]

        x = (
            x
            +
            self.time_embedding[
                :,
                :length,
                :
            ]
        )

        return self.transformer(
            x
        )


# =============================================================================
# MULTIMODAL FUSION
# =============================================================================

class MultimodalFusion(nn.Module):
    """
    Fuses:

        language
        temporal observation tokens
    """

    def __init__(
        self,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.05,
    ):

        super().__init__()

        self.language_projection = nn.Sequential(

            nn.Linear(
                256,
                hidden_dim,
            ),

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.GELU(),
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
            )
        )

    def forward(
        self,
        temporal_tokens,
        language,
    ):

        language = (
            self.language_projection(
                language
            )
            .unsqueeze(1)
        )

        tokens = torch.cat(
            [
                language,
                temporal_tokens,
            ],
            dim=1,
        )

        return self.transformer(
            tokens
        )


# =============================================================================
# ACTION CHUNK DECODER
# =============================================================================

class ActionChunkDecoder(nn.Module):
    """
    Parallel decoder for future Cartesian waypoint offsets.

    Instead of:

        current observation -> one action

    the network predicts:

        current observation -> [a_t, a_t+1, ... a_t+K]

    This follows the action-chunking direction of ACT/OpenVLA-OFT.
    """

    def __init__(
        self,
        hidden_dim=512,
        action_dim=3,
        chunk_size=ACTION_CHUNK,
        num_heads=8,
        num_layers=2,
        dropout=0.05,
    ):

        super().__init__()

        self.chunk_size = (
            chunk_size
        )

        self.query_embedding = (
            nn.Parameter(
                torch.randn(
                    1,
                    chunk_size,
                    hidden_dim,
                )
                * 0.02
            )
        )

        decoder_layer = (
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.decoder = (
            nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
            )
        )

        self.output_head = nn.Sequential(

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim // 2,
                action_dim,
            ),

            nn.Tanh(),
        )

    def forward(
        self,
        memory,
    ):

        batch_size = (
            memory.shape[0]
        )

        queries = (
            self.query_embedding
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        decoded = self.decoder(
            queries,
            memory,
        )

        return self.output_head(
            decoded
        )


# =============================================================================
# AUXILIARY PRESSURE HEAD
# =============================================================================

class PressureHead(nn.Module):
    """
    Auxiliary pressure-delta prediction.

    This is NOT the primary deployment output.

    It provides an additional learning signal from the existing
    pressure-action labels in the dataset.
    """

    def __init__(
        self,
        hidden_dim=512,
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.LayerNorm(
                hidden_dim,
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim // 2,
                3,
            ),

            nn.Tanh(),
        )

    def forward(
        self,
        feature,
    ):

        return self.head(
            feature
        )


# =============================================================================
# MAIN VLA
# =============================================================================

class SoftRobotVLA(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        action_chunk=ACTION_CHUNK,
        history_length=HISTORY_LENGTH,
        pretrained_vision=False,
        freeze_vision=False,
    ):

        super().__init__()

        self.action_chunk = (
            action_chunk
        )

        self.history_length = (
            history_length
        )

        # ---------------------------------------------------
        # ENCODERS
        # ---------------------------------------------------

        self.visual_encoder = (
            VisualEncoder(
                output_dim=256,
                pretrained=pretrained_vision,
                freeze_backbone=freeze_vision,
            )
        )

        self.language_encoder = (
            LanguageEncoder(
                vocab_size=vocab_size,
                output_dim=256,
            )
        )

        self.proprio_encoder = (
            ProprioceptionEncoder(
                input_dim=STATE_DIM,
                output_dim=256,
            )
        )

        self.trajectory_encoder = (
            TrajectoryEncoder(
                input_dim=TRAJECTORY_DIM,
                output_dim=256,
            )
        )

        # ---------------------------------------------------
        # TEMPORAL
        # ---------------------------------------------------

        self.temporal_encoder = (
            TemporalObservationEncoder(
                hidden_dim=hidden_dim,
                history_length=history_length,
            )
        )

        # ---------------------------------------------------
        # MULTIMODAL
        # ---------------------------------------------------

        self.multimodal_fusion = (
            MultimodalFusion(
                hidden_dim=hidden_dim,
            )
        )

        # ---------------------------------------------------
        # RESIDUAL REFINEMENT
        # ---------------------------------------------------

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim
                )
                for _ in range(3)
            ]
        )

        # ---------------------------------------------------
        # ACTION CHUNK
        # ---------------------------------------------------

        self.action_decoder = (
            ActionChunkDecoder(
                hidden_dim=hidden_dim,
                action_dim=3,
                chunk_size=action_chunk,
            )
        )

        # ---------------------------------------------------
        # PRESSURE AUXILIARY HEAD
        # ---------------------------------------------------

        self.pressure_head = (
            PressureHead(
                hidden_dim=hidden_dim
            )
        )

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        images,
        state,
        trajectory,
        instruction_tokens,
    ):

        """
        Parameters
        ----------
        images:
            [B,T,3,H,W]

        state:
            [B,T,9]

        trajectory:
            [B,T,14]

        instruction_tokens:
            [B,L]

        Returns
        -------
        waypoint_chunk:
            [B,K,3]

            Normalized Cartesian waypoint offsets.

        pressure_delta:
            [B,3]

            Auxiliary normalized pressure action.
        """

        batch_size, time_steps = (
            images.shape[:2]
        )

        # ---------------------------------------------------
        # IMAGE
        # ---------------------------------------------------

        images_flat = (
            images.reshape(
                batch_size * time_steps,
                *images.shape[2:],
            )
        )

        visual_flat = (
            self.visual_encoder(
                images_flat
            )
        )

        visual = (
            visual_flat.reshape(
                batch_size,
                time_steps,
                -1,
            )
        )

        # ---------------------------------------------------
        # PROPRIOCEPTION
        # ---------------------------------------------------

        proprio = (
            self.proprio_encoder(
                state.reshape(
                    batch_size * time_steps,
                    -1,
                )
            )
            .reshape(
                batch_size,
                time_steps,
                -1,
            )
        )

        # ---------------------------------------------------
        # TRAJECTORY
        # ---------------------------------------------------

        trajectory_features = (
            self.trajectory_encoder(
                trajectory.reshape(
                    batch_size * time_steps,
                    -1,
                )
            )
            .reshape(
                batch_size,
                time_steps,
                -1,
            )
        )

        # ---------------------------------------------------
        # TEMPORAL
        # ---------------------------------------------------

        temporal = (
            self.temporal_encoder(
                visual,
                proprio,
                trajectory_features,
            )
        )

        # ---------------------------------------------------
        # LANGUAGE
        # ---------------------------------------------------

        language = (
            self.language_encoder(
                instruction_tokens
            )
        )

        # ---------------------------------------------------
        # MULTIMODAL FUSION
        # ---------------------------------------------------

        fused = (
            self.multimodal_fusion(
                temporal,
                language,
            )
        )

        # ---------------------------------------------------
        # RESIDUAL REFINEMENT
        # ---------------------------------------------------

        for block in (
            self.residual_blocks
        ):

            fused = block(
                fused
            )

        # ---------------------------------------------------
        # ACTION DECODER
        # ---------------------------------------------------

        waypoint_chunk = (
            self.action_decoder(
                fused
            )
        )

        # ---------------------------------------------------
        # CURRENT STATE TOKEN
        #
        # Use the final fused token as the current control
        # context.
        # ---------------------------------------------------

        control_feature = (
            fused[:, 0]
        )

        pressure_delta = (
            self.pressure_head(
                control_feature
            )
        )

        return {
            "waypoint_chunk":
                waypoint_chunk,

            "pressure_delta":
                pressure_delta,
        }


# =============================================================================
# PARAMETER UTILITY
# =============================================================================

def count_parameters(
    model,
):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    model = SoftRobotVLA(
        vocab_size=64,
        pretrained_vision=False,
    )

    total, trainable = (
        count_parameters(
            model
        )
    )

    print(
        "Total parameters:",
        f"{total:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable:,}",
    )

    dummy_images = torch.randn(
        2,
        HISTORY_LENGTH,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    dummy_state = torch.randn(
        2,
        HISTORY_LENGTH,
        STATE_DIM,
    )

    dummy_trajectory = torch.randn(
        2,
        HISTORY_LENGTH,
        TRAJECTORY_DIM,
    )

    dummy_tokens = torch.zeros(
        2,
        MAX_INSTRUCTION_LENGTH,
        dtype=torch.long,
    )

    output = model(
        images=dummy_images,
        state=dummy_state,
        trajectory=dummy_trajectory,
        instruction_tokens=dummy_tokens,
    )

    print(
        "Waypoint chunk:",
        output[
            "waypoint_chunk"
        ].shape,
    )

    print(
        "Pressure delta:",
        output[
            "pressure_delta"
        ].shape,
    )