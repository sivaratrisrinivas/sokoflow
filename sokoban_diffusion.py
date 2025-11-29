"""
SokoFlow - Diffusion Model for Sokoban Puzzle Solving

Approach: Conditional Diffusion Policy
- Given current board state (condition), generate action sequence
- Denoising process: random actions → optimal solution actions
- Similar to "Diffusion Policy" in robotics

The model learns: p(actions | board_state) via denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ============== DIFFUSION SCHEDULE ==============

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from 'Improved DDPM' paper."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


# ============== SINUSOIDAL EMBEDDINGS ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ============== BOARD STATE ENCODER ==============

class BoardEncoder(nn.Module):
    """Encodes the 8x8 Sokoban board into a feature vector."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Input: 6 channels (one-hot: floor, wall, player, box, target, box_on_target)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, hidden_dim)
        
    def forward(self, x):
        # x: (batch, 6, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


# ============== DIFFUSION DENOISER ==============

class ActionDenoiser(nn.Module):
    """
    Denoises action sequences conditioned on board state.
    
    Input: 
        - noisy_actions: (batch, seq_len, 4) - noisy one-hot actions
        - timestep: (batch,) - diffusion timestep
        - board_state: (batch, 6, 8, 8) - current board
    
    Output:
        - denoised_actions: (batch, seq_len, 4) - predicted clean actions
    """
    def __init__(self, seq_len=20, hidden_dim=128, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Board encoder
        self.board_encoder = BoardEncoder(hidden_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action embedding (4 actions → hidden_dim)
        self.action_embed = nn.Linear(4, hidden_dim)
        
        # Position embedding for sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # Transformer for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, 4)
        
    def forward(self, noisy_actions, timestep, board_state):
        batch_size = noisy_actions.shape[0]
        
        # Encode board state: (batch, hidden_dim)
        board_feat = self.board_encoder(board_state)
        
        # Encode timestep: (batch, hidden_dim)
        time_feat = self.time_mlp(timestep)
        
        # Combine board + time as conditioning: (batch, hidden_dim)
        cond = board_feat + time_feat
        
        # Embed noisy actions: (batch, seq_len, hidden_dim)
        action_feat = self.action_embed(noisy_actions) + self.pos_embed
        
        # Prepend condition as first token
        cond = cond.unsqueeze(1)  # (batch, 1, hidden_dim)
        x = torch.cat([cond, action_feat], dim=1)  # (batch, seq_len+1, hidden_dim)
        
        # Transform
        x = self.transformer(x)
        
        # Remove condition token, project to actions
        x = x[:, 1:, :]  # (batch, seq_len, hidden_dim)
        x = self.out_proj(x)  # (batch, seq_len, 4)
        
        return x


# ============== DIFFUSION MODEL ==============

class SokobanDiffusion(nn.Module):
    """
    Full diffusion model for Sokoban action sequence generation.
    """
    def __init__(self, seq_len=20, timesteps=100, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.timesteps = timesteps
        
        # Denoiser network
        self.denoiser = ActionDenoiser(seq_len=seq_len, hidden_dim=hidden_dim)
        
        # Diffusion schedule
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to clean actions."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0, board_state):
        """Compute training loss."""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted = self.denoiser(x_noisy, t.float(), board_state)
        
        # MSE loss on noise prediction
        loss = F.mse_loss(predicted, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, board_state, temperature=1.0):
        """
        Generate action sequence via reverse diffusion.
        
        Args:
            board_state: (batch, 6, 8, 8) - encoded board
            temperature: sampling temperature
            
        Returns:
            actions: (batch, seq_len) - action indices
        """
        device = board_state.device
        batch_size = board_state.shape[0]
        
        # Start with pure noise
        x = torch.randn(batch_size, self.seq_len, 4, device=device)
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.float)
            
            # Predict noise
            predicted_noise = self.denoiser(x, t_batch, board_state)
            
            # Denoise step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x) * temperature
            else:
                noise = 0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        # Convert to action indices
        actions = torch.argmax(x, dim=-1)
        return actions

    @torch.no_grad()
    def sample_fast(self, board_state, steps=10):
        """
        Fast DDIM-style sampling with fewer steps.
        Uses deterministic updates to skip timesteps.
        """
        device = board_state.device
        batch_size = board_state.shape[0]
        
        # Start with noise
        x = torch.randn(batch_size, self.seq_len, 4, device=device)
        
        # Subsample timesteps (e.g., [90, 80, 70, ..., 10, 0] for steps=10)
        step_size = self.timesteps // steps
        timesteps = list(range(self.timesteps - 1, -1, -step_size))[:steps]
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.float)
            
            # Predict noise
            predicted_noise = self.denoiser(x, t_batch, board_state)
            
            # DDIM update (deterministic)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Predicted x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev) * predicted_noise
            
            # DDIM step
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        
        return torch.argmax(x, dim=-1)


# ============== TRAINING ==============

def state_to_tensor(grid):
    """Convert 8x8 integer grid to 6-channel one-hot tensor."""
    tensor = np.zeros((6, 8, 8), dtype=np.float32)
    for val in range(6):
        tensor[val] = (grid == val).astype(float)
    return torch.tensor(tensor)

def action_to_onehot(action_idx):
    """Convert action index to one-hot."""
    onehot = np.zeros(4, dtype=np.float32)
    onehot[action_idx] = 1.0
    return onehot

def infer_action(curr_grid, next_grid):
    """Infer which action was taken between two states."""
    curr_pos = np.argwhere(curr_grid == 2)
    next_pos = np.argwhere(next_grid == 2)
    
    if len(curr_pos) == 0 or len(next_pos) == 0:
        return 0
    
    diff = next_pos[0] - curr_pos[0]
    
    if diff[0] == -1: return 0  # UP
    if diff[0] == 1:  return 1  # DOWN
    if diff[1] == -1: return 2  # LEFT
    if diff[1] == 1:  return 3  # RIGHT
    return 0

def train_diffusion():
    """Train the diffusion model on Sokoban trajectories."""
    print("🚀 Training Sokoban Diffusion Model")
    print("-" * 50)
    
    # Load data
    data = np.load("sokoban_dataset.npy", allow_pickle=True)
    print(f"📊 Loaded {len(data)} trajectories")
    
    # Prepare training data
    seq_len = 20
    samples = []
    
    for trajectory in data:
        if len(trajectory) < 2:
            continue
            
        # Create samples from each position in trajectory
        for i in range(len(trajectory) - 1):
            start_state = trajectory[i]
            
            # Extract action sequence from this point
            actions = []
            for j in range(i, min(i + seq_len, len(trajectory) - 1)):
                action = infer_action(trajectory[j], trajectory[j + 1])
                actions.append(action)
            
            # Pad with random actions (better than always UP)
            while len(actions) < seq_len:
                actions.append(np.random.randint(0, 4))
            
            actions = actions[:seq_len]
            samples.append((start_state, actions))
    
    print(f"📝 Created {len(samples)} training samples")
    
    if len(samples) < 100:
        print("⚠️ Not enough data! Run: python sokoban_data_gen.py")
        return
    
    # Pre-convert to tensors for speed
    print("🔄 Pre-processing data...")
    all_states = np.array([state_to_tensor(s[0]).numpy() for s in samples])
    all_actions = np.array([[action_to_onehot(a) for a in s[1]] for s in samples])
    
    all_states = torch.tensor(all_states)
    all_actions = torch.tensor(all_actions)
    
    # Split into train/validation (80/20)
    split_idx = int(len(all_states) * 0.8)
    train_states = all_states[:split_idx]
    train_actions = all_actions[:split_idx]
    val_states = all_states[split_idx:]
    val_actions = all_actions[split_idx:]
    
    print(f"📊 Train samples: {len(train_states)}, Val samples: {len(val_states)}")
    
    # Create model
    model = SokobanDiffusion(seq_len=seq_len, timesteps=100, hidden_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # Training settings
    epochs = 500  # More epochs for better convergence
    batch_size = 64  # Larger batch for stability
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    print(f"🏋️ Training for {epochs} epochs, batch size {batch_size}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        # Shuffle training data
        perm = torch.randperm(len(train_states))
        train_states_shuffled = train_states[perm]
        train_actions_shuffled = train_actions[perm]
        
        total_train_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_states_shuffled), batch_size):
            states = train_states_shuffled[i:i+batch_size]
            actions = train_actions_shuffled[i:i+batch_size]
            
            loss = model.p_losses(actions, states)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        
        # ========== VALIDATION ==========
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_states), batch_size):
                states = val_states[i:i+batch_size]
                actions = val_actions[i:i+batch_size]
                
                loss = model.p_losses(actions, states)
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        
        scheduler.step()
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "sokoban_diffusion.pth")
        
        # Print progress
        if epoch % 50 == 0 or epoch < 10:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, LR = {lr:.6f}")
    
    print("-" * 50)
    print(f"✅ Training complete!")
    print(f"📊 Best train loss: {best_train_loss:.4f}")
    print(f"📊 Best val loss: {best_val_loss:.4f}")
    print(f"💾 Model saved to 'sokoban_diffusion.pth'")


if __name__ == "__main__":
    train_diffusion()

