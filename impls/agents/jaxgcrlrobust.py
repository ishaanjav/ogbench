from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, JaxGCRLValue
from flax.linen.initializers import variance_scaling

# Import the encoder architectures from train_crl_jax_brax_testing.py
class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    resnet_type: str = "resnet"

    def setup(self):
        self.lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        self.bias_init = nn.initializers.zeros
        self.normalize = nn.LayerNorm() if self.norm_type == "layer_norm" else lambda x: x
        self.activation = nn.relu if self.use_relu else nn.swish

        if self.resnet_type == "noresnet":
            self.residual_block = standard_block
        elif self.resnet_type == "resnet":
            self.residual_block = resnet_block
        elif self.resnet_type == "resnetOrig":
            self.residual_block = resnet_orig_block
        elif self.resnet_type == "identityMapping":
            self.residual_block = identity_mapping_block
        else:
            raise ValueError(f"Invalid resnet type: {self.resnet_type}")

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([s, a], axis=-1)
        
        x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        x = self.normalize(x)
        x = self.activation(x)

        num_blocks = self.network_depth // self.skip_connections
        remainder = self.network_depth % self.skip_connections
        
        for _ in range(num_blocks):
            x = self.residual_block(
                x, 
                self.network_width, 
                self.skip_connections, 
                self.normalize, 
                self.activation, 
                self.lecun_uniform, 
                self.bias_init
            )
        for i in range(remainder):
            x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
            x = self.normalize(x)
            x = self.activation(x)
        x = nn.Dense(64, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        return x

class G_encoder(nn.Module):
    norm_type: str = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 4
    use_relu: int = 0
    resnet_type: str = "resnet"

    def setup(self):
        self.lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        self.bias_init = nn.initializers.zeros
        self.normalize = nn.LayerNorm() if self.norm_type == "layer_norm" else lambda x: x
        self.activation = nn.relu if self.use_relu else nn.swish

        if self.resnet_type == "noresnet":
            self.residual_block = standard_block
        elif self.resnet_type == "resnet":
            self.residual_block = resnet_block
        elif self.resnet_type == "resnetOrig":
            self.residual_block = resnet_orig_block
        elif self.resnet_type == "identityMapping":
            self.residual_block = identity_mapping_block
        else:
            raise ValueError(f"Invalid resnet type: {self.resnet_type}")

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        x = g
        
        x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        x = self.normalize(x)
        x = self.activation(x)

        num_blocks = self.network_depth // self.skip_connections
        remainder = self.network_depth % self.skip_connections
        
        for _ in range(num_blocks):
            x = self.residual_block(
                x, 
                self.network_width, 
                self.skip_connections, 
                self.normalize, 
                self.activation, 
                self.lecun_uniform, 
                self.bias_init
            )
        for i in range(remainder):
            x = nn.Dense(self.network_width, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
            x = self.normalize(x)
            x = self.activation(x)
        x = nn.Dense(64, kernel_init=self.lecun_uniform, bias_init=self.bias_init)(x)
        return x

class JAXGCRLAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent using the JaxGCRL architecture."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive loss using JaxGCRL's approach."""
        batch_size = batch['observations'].shape[0]
        
        # Split observations into state and goal components
        state = batch['observations'][:, :self.config['obs_dim']]
        goal = batch['value_goals'][:, self.config['goal_start_idx']:self.config['goal_end_idx']]
        
        if module_name == 'critic':
            actions = batch['actions']
            sa_encoder_params = grad_params["sa_encoder"]
            g_encoder_params = grad_params["g_encoder"]
            
            sa_repr = self.network.select('sa_encoder')(state, actions, params=sa_encoder_params)
            g_repr = self.network.select('g_encoder')(goal, params=g_encoder_params)
            
            # Compute InfoNCE loss
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            I = jnp.eye(batch_size)
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
            
            # Compute additional metrics
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            
            return critic_loss, {
                'contrastive_loss': critic_loss,
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits.mean(),
            }
        else:
            return 0.0, {}

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss using JaxGCRL's approach."""
        state = batch['observations'][:, :self.config['obs_dim']]
        goal = batch['actor_goals'][:, self.config['goal_start_idx']:self.config['goal_end_idx']]
        
        # Get action distribution from actor
        dist = self.network.select('actor')(
            jnp.concatenate([state, goal], axis=-1), 
            params=grad_params
        )
        
        actions = batch['actions']
        log_prob = dist.log_prob(actions)
        
        # Compute Q-value for actions
        sa_repr = self.network.select('sa_encoder')(
            state, 
            actions,
            params=grad_params
        )
        g_repr = self.network.select('g_encoder')(
            goal,
            params=grad_params
        )
        
        q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
        
        # Compute actor loss (similar to DDPG+BC approach)
        q_loss = -q.mean()
        bc_loss = -self.config['alpha'] * log_prob.mean()
        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - actions) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent with JaxGCRL architecture."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Create the networks
        sa_encoder_def = SA_encoder(
            network_width=config['network_width'],
            network_depth=config['network_depth'],
            skip_connections=config['skip_connections'],
            use_relu=config['use_relu'],
            resnet_type=config['resnet_type']
        )
        
        g_encoder_def = G_encoder(
            network_width=config['network_width'],
            network_depth=config['network_depth'],
            skip_connections=config['skip_connections'],
            use_relu=config['use_relu'],
            resnet_type=config['resnet_type']
        )
        
        actor_def = Actor(
            action_dim=action_dim,
            network_width=config['network_width'],
            network_depth=config['network_depth'],
            skip_connections=config['skip_connections'],
            use_relu=config['use_relu'],
            resnet_type=config['resnet_type']
        )

        # Initialize the networks
        network_def = ModuleDict({
            'sa_encoder': sa_encoder_def,
            'g_encoder': g_encoder_def,
            'actor': actor_def,
        })
        
        network_tx = optax.adam(learning_rate=config['lr'])
        
        # Initialize parameters
        network_params = network_def.init(
            init_rng, 
            sa_encoder=(ex_observations, ex_actions),
            g_encoder=(ex_observations,),
            actor=(ex_observations,)
        )['params']
        
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        # Concatenate observations and goals
        if goals is not None:
            inputs = jnp.concatenate([observations, goals], axis=-1)
        else:
            inputs = observations
        
        # Get distribution from actor
        dist = self.network.select('actor')(inputs, temperature=temperature)
        
        # Sample actions
        if seed is not None:
            actions = dist.sample(seed=jax.random.PRNGKey(seed))
        else:
            actions = dist.mode()
        
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info


def get_config():
    """Get the configuration for JaxGCRL."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='jaxgcrl',
            lr=3e-4,
            batch_size=256,
            network_width=256,
            network_depth=4,
            skip_connections=4,
            use_relu=False,
            resnet_type="resnet",
            latent_dim=64,
            layer_norm=True,
            discount=0.99,
            alpha=0.1,
            discrete=False,
            obs_dim=29,  # Will be set by environment
            goal_start_idx=0,  # Will be set by environment
            goal_end_idx=2,    # Will be set by environment
            encoder=None,
            dataset_class='GCDataset',
        )
    )
    return config
