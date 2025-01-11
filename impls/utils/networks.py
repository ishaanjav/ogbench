from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
        activation_fn: Activation function to use.
        bias_init: Bias initializer to use.
    """

    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = False
    activation_fn: Any = nn.gelu
    kernel_init: Any = default_init()
    bias_init: Any = None

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                size, 
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation_fn(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


# based off the original paper
def residual_block(x, width, num_layers, layer_norm, normalize, activation, kernel_init, bias_init):
    """Process a residual block.
    
    Args:
        x: Input tensor.
        width: Width of the network layers.
        num_layers: Number of layers in this block.
        layer_norm: Whether to use layer normalization.
        normalize: Normalization function.
        activation: Activation function.
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
    """
    identity = x
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x)
        if layer_norm:
            x = normalize(x)
        if i == num_layers - 1:
            x = x + identity
        x = activation(x)
    return x

def residual_block_with_norm(x, width, num_layers, activation, kernel_init, bias_init):
    """Process a residual block with layer normalization."""
    identity = x
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x)
        if i == num_layers - 1:
            x = x + identity
        x = activation(x)
        x = nn.LayerNorm()(x)
    return x

def residual_block_no_norm(x, width, num_layers, activation, kernel_init, bias_init):
    """Process a residual block without layer normalization."""
    identity = x
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x)
        if i == num_layers - 1:
            x = x + identity
        x = activation(x)
    return x

def sequential_no_norm(x, width, num_layers, activation, kernel_init, bias_init):
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = activation(x)
    return x

def sequential_with_norm(x, width, num_layers, activation, kernel_init, bias_init):
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = activation(x)
        x = nn.LayerNorm()(x)
    return x

class ResNet(nn.Module):
    """Multi-layer perceptron with residual connections."""

    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = False
    skip_connection_frequency: int = 4
    activation_fn: Any = nn.gelu
    kernel_init: Any = default_init()
    bias_init: Any = None

    def setup(self):
        self.residual_block = residual_block_with_norm if self.layer_norm else residual_block_no_norm
        self.sequential = sequential_with_norm if self.layer_norm else sequential_no_norm

    @nn.compact
    def __call__(self, x):
        # Initial layer
        x = nn.Dense(
            self.hidden_dims[0], 
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = self.activation_fn(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)

        num_blocks = (len(self.hidden_dims) ) // self.skip_connection_frequency
        remainder = (len(self.hidden_dims)) % self.skip_connection_frequency
        
        # Process blocks
        for _ in range(num_blocks):
            x = self.residual_block(
                x=x,
                width=self.hidden_dims[0],
                num_layers=self.skip_connection_frequency,
                activation=self.activation_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )

        # Process remainder layers
        x = self.sequential(x, self.hidden_dims[0], remainder, self.activation_fn, self.kernel_init, self.bias_init)

        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)

lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros
# # original code
# def residual_block(x, width, normalize, activation, num_layers):
#     identity = x
#     # Apply num_layers dense layers
#     for _ in range(num_layers):
#         x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = activation(x)
#     x = x + identity  # Skip connection
#     return x

# no resnet
def standard_block(x, width, num_layers, normalize, activation, lecun_uniform, bias_init):
    for _ in range(num_layers):
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
    return x

# our (JaxGCRL) implementation of the resnet block
def resnet_block(x, width, num_layers, normalize, activation, lecun_uniform, bias_init):
    identity = x
    for _ in range(num_layers):
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
    return x + identity

# from the original paper
def resnet_orig_block(x, width, num_layers, normalize, activation, kernel_init):
    identity = x
    for i in range(num_layers):
        x = nn.Dense(width, kernel_init=kernel_init)(x)
        x = normalize(x)
        if i == num_layers - 1:
            x = x + identity
        x = activation(x)
    return x

# from the identity mapping paper
def identity_mapping_block(x, width, num_layers, normalize, activation, lecun_uniform, bias_init):
    identity = x
    for i in range(num_layers):
        x = normalize(x)
        x = activation(x)
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        if i == num_layers - 1:
            x = x + identity
    x = normalize(x)
    x = activation(x)
    return x

def standard_layer(x, width, normalize, activation, lecun_uniform, bias_init):
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    return x

def resnet_layer(x, width, normalize, activation, lecun_uniform, bias_init):
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    return x

def identity_mapping_layer(x, width, normalize, activation, lecun_uniform, bias_init):
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    return x

# From JaxGCRL implementation
class Actor(nn.Module):
    """Goal-conditioned actor with customizable ResNet architecture.
    
    Attributes:
        action_dim: Action dimension.
        hidden_dims: Hidden layer dimensions (replaced by network_width & depth).
        network_width: Width of network layers.
        network_depth: Total number of layers.
        skip_connection_frequency: Number of layers per residual block.
        use_relu: Whether to use ReLU (False uses swish).
        resnet_type: Type of residual connections ("resnet", "noresnet", "resnetOrig", "identityMapping").
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """
    
    action_dim: int
    hidden_dims: Sequence[int] = None  # Kept for compatibility but unused
    network_width: int = 1024
    network_depth: int = 4
    skip_connection_frequency: int = 4
    use_relu: bool = False
    resnet_type: str = "resnet"
    log_std_min: float = -5
    log_std_max: float = 2
    state_dependent_std: bool = False
    const_std: bool = True

    def setup(self):
        self.lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        self.bias_init = nn.initializers.zeros
        self.normalize = nn.LayerNorm()
        self.activation = nn.relu if self.use_relu else nn.swish

        # Select residual block type
        if self.resnet_type == "noresnet":
            self.residual_block = standard_block
            self.layer = standard_layer
        elif self.resnet_type == "resnet":
            self.residual_block = resnet_block
            self.layer = resnet_layer
        elif self.resnet_type == "resnetOrig":
            self.residual_block = resnet_orig_block
            self.layer = resnet_layer
        elif self.resnet_type == "identityMapping":
            self.residual_block = identity_mapping_block
            self.layer = identity_mapping_layer
        else:
            raise ValueError(f"Invalid resnet type: {self.resnet_type}")

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        """Return the action distribution.
        
        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        # NOTE: The original GCActor uses the GC_Encoder (specifically concat_encoder) as defined encoders.py
        #   We will be using our own, original Actor network which "encodes" the state + goal in the first layer itself
        # if self.gc_encoder is not None:
        #     x = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        # else:
        #     inputs = [observations]
        #     if goals is not None:
        #         inputs.append(goals)
        #     x = jnp.concatenate(inputs, axis=-1)

        # Concatenate the state and goal
        x = jnp.concatenate([observations, goals], axis=-1)

        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        x = self.normalize(x)
        x = self.activation(x)

        # Use skip_connection_frequency to determine layers per block
        num_blocks = self.network_depth // self.skip_connection_frequency
        remainder = self.network_depth % self.skip_connection_frequency
        
        # for _ in range(num_blocks):
        #     x = self.residual_block(
        #         x, 
        #         self.network_width, 
        #         self.skip_connection_frequency, 
        #         self.normalize, 
        #         self.activation, 
        #         self.lecun_uniform, 
        #         self.bias_init
        #     )
        identity = x
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
            x = self.normalize(x)
            if i == self.network_depth - 1:
                x = x + identity
            x = self.activation(x)

        # Remainder layers
        # TODO: this should follow the same patterns of the 4 above
        for i in range(remainder):
            x = self.layer(x, self.network_width, self.normalize, self.activation, self.lecun_uniform, self.bias_init)

        mean = nn.Dense(self.action_dim, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        log_std = nn.Dense(self.action_dim, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        distribution = distrax.MultivariateNormalDiag(
            loc=mean, 
            scale_diag=jnp.exp(log_std) * temperature
        )
        
        return distribution
    
class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        use_resnet: Whether to use ResNet instead of MLP.
        skip_connection_frequency: Number of layers per residual block.
        activation_fn: Activation function to use.
        kernel_init: Kernel initializer to use.
        bias_init: Bias initializer to use.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    use_resnet: bool = False
    skip_connection_frequency: int = 4
    activation_fn: Any = nn.gelu
    kernel_init: Any = default_init()
    bias_init: Any = None

    def setup(self):
        # Choose network architecture based on use_resnet flag
        if self.use_resnet:
            self.actor_net = ResNet(
                self.hidden_dims, 
                activate_final=True,
                skip_connection_frequency=self.skip_connection_frequency,
                activation_fn=self.activation_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                layer_norm=False,
            )
        else:
            self.actor_net = MLP(
                self.hidden_dims, 
                activate_final=True,
                activation_fn=self.activation_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            
        self.mean_net = nn.Dense(
            self.action_dim, 
            kernel_init=default_init(self.final_fc_init_scale),
            bias_init=self.bias_init,
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, 
                kernel_init=default_init(self.final_fc_init_scale),
                bias_init=self.bias_init,
            )
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None: # this is typically only for visual environments
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else: 
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class SA_encoder(nn.Module):
    """State-Action encoder with ResNet architecture."""
    
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connection_frequency: int = 4
    use_relu: int = 0
    resnet_type: str = "resnet"
    latent_dim: int = 64

    def setup(self):
        self.lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        self.bias_init = nn.initializers.zeros
        self.normalize = nn.LayerNorm() if self.norm_type == "layer_norm" else lambda x: x
        self.activation = nn.relu if self.use_relu else nn.swish

        # Select residual block type
        if self.resnet_type == "noresnet":
            self.residual_block = standard_block
            self.layer = standard_layer
        elif self.resnet_type == "resnet":
            self.residual_block = resnet_block
            self.layer = resnet_layer
        elif self.resnet_type == "resnetOrig":
            self.residual_block = resnet_orig_block
            self.layer = resnet_layer
        elif self.resnet_type == "identityMapping":
            self.residual_block = identity_mapping_block
            self.layer = identity_mapping_layer
        else:
            raise ValueError(f"Invalid resnet type: {self.resnet_type}")

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        # Ensure inputs are arrays and concatenate them
        observations = jnp.asarray(observations)
        actions = jnp.asarray(actions)
        x = jnp.concatenate([observations, actions], axis=-1)
        
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        x = self.normalize(x)
        x = self.activation(x)

        # Process blocks
        num_blocks = self.network_depth // self.skip_connection_frequency
        remainder = self.network_depth % self.skip_connection_frequency
        
        # for _ in range(num_blocks):
        #     x = self.residual_block(
        #         x, 
        #         self.network_width, 
        #         self.skip_connection_frequency, 
        #         self.normalize, 
        #         self.activation, 
        #         self.lecun_uniform, 
        #         self.bias_init
        #     )
        identity = x
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
            x = self.normalize(x)
            if i == self.network_depth - 1:
                x = x + identity
            x = self.activation(x)

        # Process remainder layers
        for _ in range(remainder):
            x = self.layer(
                x, 
                self.network_width, 
                self.normalize, 
                self.activation, 
                self.lecun_uniform, 
                self.bias_init
            )

        # Final projection to latent dimension
        x = nn.Dense(self.latent_dim, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        return x

class G_encoder(nn.Module):
    """Goal encoder with ResNet architecture."""
    
    norm_type: str = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connection_frequency: int = 4
    use_relu: int = 0
    resnet_type: str = "resnet"  # Options: "resnet", "noresnet", "resnetOrig", "identityMapping"
    latent_dim: int = 64

    def setup(self):
        self.lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        self.bias_init = nn.initializers.zeros
        self.normalize = nn.LayerNorm() if self.norm_type == "layer_norm" else lambda x: x
        self.activation = nn.relu if self.use_relu else nn.swish

        # Select residual block type
        if self.resnet_type == "noresnet":
            self.residual_block = standard_block
            self.layer = standard_layer
        elif self.resnet_type == "resnet":
            self.residual_block = resnet_block
            self.layer = resnet_layer
        elif self.resnet_type == "resnetOrig":
            self.residual_block = resnet_orig_block
            self.layer = resnet_layer
        elif self.resnet_type == "identityMapping":
            self.residual_block = identity_mapping_block
            self.layer = identity_mapping_layer
        else:
            raise ValueError(f"Invalid resnet type: {self.resnet_type}")

    @nn.compact
    def __call__(self, goals: jnp.ndarray) -> jnp.ndarray:
        # Ensure input is an array
        x = jnp.asarray(goals)
        
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        x = self.normalize(x)
        x = self.activation(x)

        # Process blocks
        num_blocks = self.network_depth // self.skip_connection_frequency
        remainder = self.network_depth % self.skip_connection_frequency
        
        # for _ in range(num_blocks):
        #     x = self.residual_block(
        #         x, 
        #         self.network_width, 
        #         self.skip_connection_frequency, 
        #         self.normalize, 
        #         self.activation, 
        #         self.lecun_uniform, 
        #         self.bias_init
        #     )
        identity = x
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
            x = self.normalize(x)
            if i == self.network_depth - 1:
                x = x + identity
            x = self.activation(x)

        # Process remainder layers
        for _ in range(remainder):
            x = self.layer(
                x, 
                self.network_width, 
                self.normalize, 
                self.activation, 
                self.lecun_uniform, 
                self.bias_init
            )

        # Final projection to latent dimension
        x = nn.Dense(self.latent_dim, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        return x


class JaxGCRLValue(nn.Module):
    """Goal-conditioned bilinear value/critic function using SA_encoder and G_encoder.

    Attributes:
        hidden_dims: Hidden layer dimensions (kept for compatibility but unused).
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value.
        network_width: Width of network layers for both encoders.
        network_depth: Total number of layers for both encoders.
        skip_connection_frequency: Number of layers per residual block for both encoders.
        use_relu: Whether to use ReLU (False uses swish) for both encoders.
        resnet_type: Type of residual connections for both encoders.
        latent_dim: Dimension of the latent space for both encoders.
    """

    ensemble: bool = True
    value_exp: bool = False
    # Network parameters for encoders
    network_width: int = 1024
    network_depth: int = 4
    skip_connection_frequency: int = 4
    use_relu: bool = False
    resnet_type: str = "resnet"
    embedding_dim: int = 64
    layer_norm: bool = True

    def setup(self) -> None:
        encoder_module = lambda: SA_encoder(
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connection_frequency=self.skip_connection_frequency,
            use_relu=self.use_relu,
            resnet_type=self.resnet_type,
            latent_dim=self.embedding_dim  # Pass through the single embedding_dim
        )
        goal_encoder_module = lambda: G_encoder(
            network_width=self.network_width,
            network_depth=self.network_depth,
            skip_connection_frequency=self.skip_connection_frequency,
            use_relu=self.use_relu,
            resnet_type=self.resnet_type,
            latent_dim=self.embedding_dim  # Pass through the single embedding_dim
        )

        if self.ensemble:
            self.phi = nn.vmap(
                encoder_module,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                out_axes=0,
                axis_size=2
            )
            self.psi = nn.vmap(
                goal_encoder_module,
                variable_axes={'params': 0},
                split_rngs={'params': True},
                in_axes=None,
                out_axes=0,
                axis_size=2
            )
        else:
            self.phi = encoder_module()
            self.psi = goal_encoder_module()

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if actions is None:
            raise ValueError("Actions must be provided to compute phi.")
        
        observations = jnp.asarray(observations)
        goals = jnp.asarray(goals)
        actions = jnp.asarray(actions)
        
        phi = self.phi(observations, actions)
        psi = self.psi(goals)

        # Note: Both encoders output 64-dimensional vectors, so we use that as embedding_dim
        v = (phi * psi / jnp.sqrt(self.embedding_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v

class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
        use_resnet: Whether to use ResNet instead of MLP.
        skip_connection_frequency: Number of layers per residual block.
        activation_fn: Activation function to use.
        kernel_init: Kernel initializer to use.
        bias_init: Bias initializer to use.
    """

    hidden_dims: Sequence[int]
    latent_dim: int = 64
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    use_resnet: bool = False
    skip_connection_frequency: int = 4
    activation_fn: Any = nn.gelu
    kernel_init: Any = default_init()
    bias_init: Any = None

    def setup(self) -> None:
        # Choose base network architecture
        base_module = ResNet if self.use_resnet else MLP
        mlp_module = base_module
        
        if self.ensemble:
            mlp_module = ensemblize(base_module, 2)

        self.phi = mlp_module(
            (*self.hidden_dims, self.latent_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            skip_connection_frequency=self.skip_connection_frequency,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.psi = mlp_module(
            (*self.hidden_dims, self.latent_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            skip_connection_frequency=self.skip_connection_frequency,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @nn.compact
    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        # 1. Optional encoding of visual inputs
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        # 2. Combine state and action if action is provided
        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        # 3. Process through the state-action and goal encoding networks
        phi = self.phi(phi_inputs)
        phi = nn.Dense(
            self.latent_dim, 
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(phi)

        psi = self.psi(goals)
        psi = nn.Dense(
            self.latent_dim, 
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(psi)

        # 4. Compute the value function
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v
