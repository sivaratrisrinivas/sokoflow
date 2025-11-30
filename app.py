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

def to_python_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    This bypasses any Flask/NumPy version incompatibilities.
    """
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_types(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return to_python_types(obj.tolist())
    elif isinstance(obj, np.generic):
        # Handles all numpy scalars: np.bool_, np.int64, np.float32, etc.
        return obj.item()
    return obj

ACTION_MAP = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
ACTION_DELTAS = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}

# --- LOAD MODEL ---
diffusion_model = SokobanDiffusion(seq_len=20, timesteps=100, hidden_dim=128)

if os.path.exists("sokoban_diffusion.pth"):
    try:
        # map_location='cpu' ensures it loads even if trained on GPU
        diffusion_model.load_state_dict(torch.load("sokoban_diffusion.pth", map_location='cpu', weights_only=True))
        diffusion_model.eval()
        torch.set_grad_enabled(False)
        print("🎨 Diffusion model loaded!")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print("⚠️ Train model first: python sokoban_diffusion.py")

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
    return int(np.count_nonzero(grid == 3)) == 0

def diffusion_solve_fast(grid, targets, max_iters=20):
    if not os.path.exists("sokoban_diffusion.pth"):
        return None
    
    current_grid = grid.copy()
    current_pos = tuple(map(int, np.argwhere(current_grid == 2)[0]))
    solution = []
    visited = {current_grid.tobytes()}
    
    for iteration in range(max_iters):
        if is_solved(current_grid):
            return solution
        
        # Batch sample
        state_tensor = state_to_tensor(current_grid).unsqueeze(0)
        batch_state = state_tensor.repeat(4, 1, 1, 1)
        
        with torch.inference_mode():
            all_actions = diffusion_model.sample_fast(batch_state, steps=10)
        
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
            
            boxes_done = np.count_nonzero(test_grid == 5)
            progress = len(test_solution) + boxes_done * 2
            
            if progress > best_progress:
                best_progress = progress
                best_result = (test_grid, test_pos, test_solution, test_visited)
        
        if best_result and best_result[2]:
            current_grid, current_pos, new_moves, visited = best_result
            solution.extend(new_moves)
        else:
            # Random escape
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
    return jsonify(to_python_types({'boxes': boxes}))

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global solution_path, solution_index
    solution_path = []
    solution_index = 0
    
    difficulty = request.json.get('difficulty', 20)
    
    # Try to generate and solve
    for attempt in range(5):
        env.reset_solved()
        for _ in range(difficulty):
            env.step_reverse()
        
        solution_path = diffusion_solve_fast(env.grid, env.targets)
        if solution_path:
            break
        difficulty = max(8, difficulty - 4)
    
    response = {
        'grid': env.grid,
        'targets': env.targets,
        'solvable': bool(solution_path is not None),
        'moves': len(solution_path) if solution_path else 0
    }
    # Sanitize response before JSON serialization
    return jsonify(to_python_types(response))

@app.route('/api/solve_step', methods=['POST'])
def solve_step():
    global solution_index
    
    if not solution_path:
        return jsonify(to_python_types({
            'action': 'GIVE_UP',
            'grid': env.grid,
            'solved': False,
            'gave_up': True,
            'steps': 0,
            'message': 'Diffusion failed'
        }))
    
    if solution_index >= len(solution_path):
        return jsonify(to_python_types({
            'action': 'DONE',
            'grid': env.grid,
            'solved': is_solved(env.grid),
            'steps': solution_index,
            'total': len(solution_path)
        }))
    
    action = solution_path[solution_index]
    solution_index += 1
    env.step(action)
    
    response = {
        'action': action,
        'grid': env.grid,
        'solved': is_solved(env.grid),
        'steps': solution_index,
        'total': len(solution_path)
    }
    return jsonify(to_python_types(response))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 SokoFlow")
    print(f"   http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)