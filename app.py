"""
SokoFlow - Diffusion Sokoban Solver

A neural diffusion model that solves Sokoban puzzles.
Flow = The denoising process that transforms random moves → optimal solution.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import torch
import numpy as np
import os
import json

from sokoban_engine import SokobanEnv
from sokoban_diffusion import SokobanDiffusion, state_to_tensor

app = Flask(__name__)
CORS(app)

# Custom JSON provider to handle numpy types (Flask 2.3+)
try:
    from flask.json.provider import DefaultJSONProvider
    
    class NumpyJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    app.json = NumpyJSONProvider(app)
except ImportError:
    # Fallback for older Flask versions
    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    app.json_encoder = NumpyJSONEncoder

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    # Handle numpy scalars first (before checking for arrays)
    if isinstance(obj, np.ndarray):
        # Convert array to list first, then recursively convert elements
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    # Handle numpy integer types
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Handle numpy float types
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    # Handle numpy bool - CRITICAL: must check before Python bool
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # For any other numpy type, try to convert using .item()
    elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):  # numpy scalars have both
        try:
            return obj.item()
        except:
            return obj
    # Python native types pass through
    return obj

ACTION_MAP = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
ACTION_DELTAS = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}

# --- LOAD & OPTIMIZE DIFFUSION MODEL ---
diffusion_model = None
MODEL_LOADED = False

def load_model():
    """Load the diffusion model."""
    global diffusion_model, MODEL_LOADED
    
    model_path = "sokoban_diffusion.pth"
    if os.path.exists(model_path):
        try:
            diffusion_model = SokobanDiffusion(seq_len=20, timesteps=100, hidden_dim=128)
            diffusion_model.load_state_dict(torch.load(model_path, weights_only=True))
            diffusion_model.eval()
            
            # Optimization 1: Disable gradient computation globally
            torch.set_grad_enabled(False)
            
            # Optimization 2: Use inference mode (faster than no_grad)
            # Optimization 3: Compile model for faster inference (PyTorch 2.0+)
            try:
                diffusion_model = torch.compile(diffusion_model, mode="reduce-overhead")
                print("🚀 Model compiled with torch.compile!")
            except:
                pass
            
            MODEL_LOADED = True
            print("🎨 Diffusion model loaded & optimized!")
            return True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            diffusion_model = None
            MODEL_LOADED = False
            return False
    else:
        # Don't print warning if TRAIN_ON_STARTUP is enabled (Railway will train automatically)
        if os.environ.get('TRAIN_ON_STARTUP', 'false').lower() != 'true':
            print(f"⚠️ Model file '{model_path}' not found. Upload model file or enable auto-training.")
        return False

# Try to load model
load_model()

# --- GLOBAL STATE ---
env = SokobanEnv(num_boxes=3)
solution_path = []
solution_index = 0

def is_valid_move(grid, targets, pos, action):
    dr, dc = ACTION_DELTAS[action]
    pr, pc = pos
    nr, nc = pr + dr, pc + dc
    
    if not (0 <= nr < 8 and 0 <= nc < 8):
        return None, None
    
    ng = grid.copy()
    
    if ng[nr, nc] == 1:
        return None, None
    
    if ng[nr, nc] in [0, 4]:
        ng[pr, pc] = 4 if targets[pr, pc] else 0
        ng[nr, nc] = 2
        return ng, (nr, nc)
    
    if ng[nr, nc] in [3, 5]:
        nnr, nnc = nr + dr, nc + dc
        if 0 <= nnr < 8 and 0 <= nnc < 8 and ng[nnr, nnc] in [0, 4]:
            ng[pr, pc] = 4 if targets[pr, pc] else 0
            ng[nr, nc] = 2
            ng[nnr, nnc] = 5 if targets[nnr, nnc] else 3
            return ng, (nr, nc)
    
    return None, None

def is_solved(grid):
    # Ensure we return Python bool, not numpy bool
    result = np.count_nonzero(grid == 3) == 0
    return bool(result) if isinstance(result, (np.bool_, bool)) else bool(result)

def diffusion_solve_fast(grid, targets, max_iters=20):
    """
    Optimized diffusion solving:
    - DDIM sampling (10 steps instead of 100)  
    - Batch sampling (try 4 sequences, pick best)
    - Early stopping when solved
    """
    if not MODEL_LOADED or diffusion_model is None:
        return None  # Model not available
    
    current_grid = grid.copy()
    current_pos = tuple(map(int, np.argwhere(current_grid == 2)[0]))
    solution = []
    visited = {current_grid.tobytes()}
    
    for iteration in range(max_iters):
        if is_solved(current_grid):
            return solution
        
        # Batch sample: generate 4 action sequences in parallel
        state_tensor = state_to_tensor(current_grid).unsqueeze(0)
        batch_state = state_tensor.repeat(4, 1, 1, 1)  # (4, 6, 8, 8)
        
        with torch.inference_mode():
            all_actions = diffusion_model.sample_fast(batch_state, steps=10)  # (4, 20)
        
        # Try each sequence, pick the one that makes most progress
        best_progress = 0
        best_result = None
        
        for seq_idx in range(4):
            test_grid = current_grid.copy()
            test_pos = current_pos
            test_solution = []
            test_visited = visited.copy()
            
            actions = all_actions[seq_idx].tolist()
            
            for action_idx in actions:
                if is_solved(test_grid):
                    # Found solution! Return immediately
                    return solution + test_solution
                
                action = ACTION_MAP[action_idx]
                new_grid, new_pos = is_valid_move(test_grid, targets, test_pos, action)
                
                if new_grid is not None:
                    h = new_grid.tobytes()
                    if h not in test_visited:
                        test_visited.add(h)
                        test_grid = new_grid
                        test_pos = new_pos
                        test_solution.append(action)
            
            # Score: moves made + bonus for boxes on targets
            boxes_done = np.count_nonzero(test_grid == 5)
            progress = len(test_solution) + boxes_done * 2
            
            if progress > best_progress:
                best_progress = progress
                best_result = (test_grid, test_pos, test_solution, test_visited)
        
        # Apply best sequence
        if best_result and best_result[2]:
            current_grid, current_pos, new_moves, visited = best_result
            solution.extend(new_moves)
        else:
            # Stuck - try random escape
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                new_grid, new_pos = is_valid_move(current_grid, targets, current_pos, action)
                if new_grid is not None and new_grid.tobytes() not in visited:
                    visited.add(new_grid.tobytes())
                    current_grid = new_grid
                    current_pos = new_pos
                    solution.append(action)
                    break
    
    return solution if is_solved(current_grid) else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/set_boxes', methods=['POST'])
def set_boxes():
    global env
    boxes = max(2, min(4, request.json.get('boxes', 3)))
    env = SokobanEnv(num_boxes=boxes)
    return jsonify({'boxes': boxes})

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global solution_path, solution_index
    solution_path = []
    solution_index = 0
    
    difficulty = request.json.get('difficulty', 20)
    
    # Generate puzzle and solve with diffusion
    for attempt in range(5):
        env.reset_solved()
        for _ in range(difficulty):
            env.step_reverse()
        
        solution_path = diffusion_solve_fast(env.grid, env.targets)
        
        if solution_path:
            break
        
        # Try easier puzzle
        difficulty = max(8, difficulty - 4)
    
    # Convert numpy arrays to Python lists - ensure all types are native Python
    grid_list = env.grid.astype(int).tolist()
    targets_list = env.targets.astype(int).tolist()
    
    return jsonify({
        'grid': grid_list,
        'targets': targets_list,
        'solvable': bool(solution_path is not None),
        'moves': int(len(solution_path) if solution_path else 0)
    })

@app.route('/api/solve_step', methods=['POST'])
def solve_step():
    global solution_index
    
    if not solution_path:
        grid_list = env.grid.astype(int).tolist()
        
        return jsonify({
            'action': 'GIVE_UP',
            'grid': grid_list,
            'solved': False,
            'gave_up': True,
            'steps': 0,
            'message': 'Diffusion failed'
        })
    
    if solution_index >= len(solution_path):
        grid_list = convert_numpy_types(env.grid.tolist())
        solved = bool(is_solved(env.grid))
        
        return jsonify({
            'action': 'DONE',
            'grid': grid_list,
            'solved': solved,
            'steps': int(solution_index),
            'total': int(len(solution_path))
        })
    
    action = solution_path[solution_index]
    solution_index += 1
    env.step(action)
    
    # Ensure all types are Python native - convert array to int first, then to list
    grid_list = env.grid.astype(int).tolist()
    solved = bool(is_solved(env.grid))
    
    return jsonify({
        'action': str(action),
        'grid': grid_list,
        'solved': solved,
        'steps': int(solution_index),
        'total': int(len(solution_path))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 SokoFlow")
    print(f"   http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
