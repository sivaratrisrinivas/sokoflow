"""
Enhanced Sokoban Dataset Generator

Generates diverse training data:
- Multiple difficulty levels
- Different box counts (2-4)
- More episodes for better coverage
"""

import numpy as np
import random
from collections import deque

FLOOR, WALL, PLAYER, BOX, TARGET, BOX_TARGET = 0, 1, 2, 3, 4, 5
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


class SokobanGen:
    def __init__(self, width=8, height=8, num_boxes=3):
        self.width = width
        self.height = height
        self.num_boxes = num_boxes
        self.grid = np.zeros((height, width), dtype=int)
        self.targets = np.zeros((height, width), dtype=bool)
        self.player_pos = (0, 0)

    def reset_solved(self):
        self.grid.fill(FLOOR)
        self.targets.fill(False)
        
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        self.player_pos = self._get_random_free_pos()
        self.grid[self.player_pos] = PLAYER
        
        for _ in range(self.num_boxes):
            r, c = self._get_random_free_pos()
            self.grid[r, c] = BOX_TARGET
            self.targets[r, c] = True
            
        return self.grid.copy()

    def _get_random_free_pos(self):
        while True:
            r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
            if self.grid[r, c] == FLOOR and not self.targets[r, c]:
                return (r, c)

    def step_reverse(self):
        """
        Improved reverse step that prioritizes pulling boxes off targets.
        """
        # First, try to find a box on a target and pull it off
        boxes_on_targets = np.argwhere(self.grid == BOX_TARGET)
        if len(boxes_on_targets) > 0:
            # Try to pull a box off a target (higher priority)
            directions = list(ACTIONS.keys())
            random.shuffle(directions)
            
            for action in directions:
                dr, dc = ACTIONS[action]
                pr, pc = self.player_pos
                br, bc = pr - dr, pc - dc  # Position behind player
                nr, nc = pr + dr, pc + dc  # Position in front of player
                
                # Check if there's a box on target behind us and we can move forward
                if (0 < br < self.height-1 and 0 < bc < self.width-1 and
                    0 < nr < self.height-1 and 0 < nc < self.width-1 and
                    self.grid[br, bc] == BOX_TARGET and
                    self.grid[nr, nc] != WALL and 
                    self.grid[nr, nc] not in [BOX, BOX_TARGET]):
                    # Force pull: move player forward, pull box to player's old position
                    self._move_entity(pr, pc, nr, nc, is_player=True)
                    self._move_entity(br, bc, pr, pc, is_player=False)
                    return self.grid.copy(), True
        
        # Otherwise, do a normal random reverse move
        action = random.choice(list(ACTIONS.keys()))
        dr, dc = ACTIONS[action]
        
        pr, pc = self.player_pos
        nr, nc = pr + dr, pc + dc
        br, bc = pr - dr, pc - dc

        if not (0 < nr < self.height-1 and 0 < nc < self.width-1):
            return self.grid.copy(), False

        if self.grid[nr, nc] == WALL:
            return self.grid.copy(), False
            
        if self.grid[nr, nc] in [BOX, BOX_TARGET]:
            return self.grid.copy(), False

        has_box_behind = False
        if 0 < br < self.height-1 and 0 < bc < self.width-1:
            if self.grid[br, bc] in [BOX, BOX_TARGET]:
                has_box_behind = True

        # Higher pull probability for boxes on targets (90%), regular boxes (70%)
        if has_box_behind:
            if self.grid[br, bc] == BOX_TARGET:
                should_pull = (random.random() < 0.9)  # 90% chance
            else:
                should_pull = (random.random() < 0.7)  # 70% chance
        else:
            should_pull = False

        if should_pull:
            self._move_entity(pr, pc, nr, nc, is_player=True)
            self._move_entity(br, bc, pr, pc, is_player=False)
        else:
            self._move_entity(pr, pc, nr, nc, is_player=True)
            
        return self.grid.copy(), True

    def _move_entity(self, r, c, nr, nc, is_player):
        was_target = self.targets[r, c]
        self.grid[r, c] = TARGET if was_target else FLOOR
        
        is_new_spot_target = self.targets[nr, nc]
        
        if is_player:
            self.grid[nr, nc] = PLAYER
            self.player_pos = (nr, nc)
        else:
            self.grid[nr, nc] = BOX_TARGET if is_new_spot_target else BOX


def is_valid_move(grid, targets, pos, action_idx):
    """Check if move is valid, return new grid and pos."""
    dr, dc = ACTIONS[action_idx]
    pr, pc = pos
    nr, nc = pr + dr, pc + dc
    
    if not (0 <= nr < 8 and 0 <= nc < 8):
        return None, None
    
    ng = grid.copy()
    
    if ng[nr, nc] == WALL:
        return None, None
    
    if ng[nr, nc] in [FLOOR, TARGET]:
        ng[pr, pc] = TARGET if targets[pr, pc] else FLOOR
        ng[nr, nc] = PLAYER
        return ng, (nr, nc)
    
    if ng[nr, nc] in [BOX, BOX_TARGET]:
        nnr, nnc = nr + dr, nc + dc
        if 0 <= nnr < 8 and 0 <= nnc < 8 and ng[nnr, nnc] in [FLOOR, TARGET]:
            ng[pr, pc] = TARGET if targets[pr, pc] else FLOOR
            ng[nr, nc] = PLAYER
            ng[nnr, nnc] = BOX_TARGET if targets[nnr, nnc] else BOX
            return ng, (nr, nc)
    
    return None, None


def bfs_solve(grid, targets, max_nodes=30000):
    """BFS to find optimal solution path."""
    start = grid.copy()
    start_pos = tuple(map(int, np.argwhere(start == 2)[0]))
    
    queue = deque([(start, start_pos, [])])
    visited = {start.tobytes()}
    
    while queue and len(visited) < max_nodes:
        g, pos, path = queue.popleft()
        
        if np.count_nonzero(g == 3) == 0:
            return path
        
        for action_idx in range(4):
            ng, new_pos = is_valid_move(g, targets, pos, action_idx)
            if ng is None:
                continue
            
            h = ng.tobytes()
            if h not in visited:
                visited.add(h)
                queue.append((ng, new_pos, path + [action_idx]))
    
    return None


def generate_trajectory(env, scramble_steps):
    """Generate a single (scrambled_state, solution_actions) trajectory."""
    env.reset_solved()
    
    # Scramble
    for _ in range(scramble_steps):
        env.step_reverse()
    
    # Find solution
    solution = bfs_solve(env.grid, env.targets)
    
    if solution is None:
        return None
    
    # Create trajectory: execute solution and record each (state, action)
    trajectory = []
    current_grid = env.grid.copy()
    current_pos = tuple(map(int, np.argwhere(current_grid == 2)[0]))
    
    for action_idx in solution:
        trajectory.append((current_grid.copy(), action_idx))
        new_grid, new_pos = is_valid_move(current_grid, env.targets, current_pos, action_idx)
        if new_grid is not None:
            current_grid = new_grid
            current_pos = new_pos
    
    return trajectory


def generate_dataset(num_episodes=500, output_file="sokoban_dataset.npy"):
    """
    Generate diverse training dataset with harder puzzles.
    
    Creates trajectories with:
    - Different box counts (2, 3, 4)
    - Higher scramble levels (20-60 steps) for harder puzzles
    - More 4-box puzzles for complexity
    """
    all_trajectories = []
    
    configs = [
        # Easy puzzles (for learning basics)
        (2, 20, 60),   # 2 boxes, 20 scramble steps, 60 episodes
        (2, 30, 50),   # 2 boxes, 30 scramble steps, 50 episodes
        (2, 40, 40),   # 2 boxes, 40 scramble steps, 40 episodes
        
        # Medium puzzles (main training data)
        (3, 20, 80),   # 3 boxes, 20 scramble steps, 80 episodes
        (3, 30, 70),   # 3 boxes, 30 scramble steps, 70 episodes
        (3, 40, 60),   # 3 boxes, 40 scramble steps, 60 episodes
        (3, 50, 40),   # 3 boxes, 50 scramble steps, 40 episodes
        
        # Hard puzzles (4 boxes, more scramble)
        (4, 20, 40),   # 4 boxes, 20 scramble steps, 40 episodes
        (4, 30, 35),   # 4 boxes, 30 scramble steps, 35 episodes
        (4, 40, 30),   # 4 boxes, 40 scramble steps, 30 episodes
        (4, 50, 25),   # 4 boxes, 50 scramble steps, 25 episodes
        (4, 60, 20),   # 4 boxes, 60 scramble steps, 20 episodes
    ]
    
    total = sum(c[2] for c in configs)
    generated = 0
    
    print(f"🎯 Generating {total} trajectories...")
    print("-" * 50)
    
    for num_boxes, scramble_steps, count in configs:
        env = SokobanGen(num_boxes=num_boxes)
        success = 0
        attempts = 0
        
        while success < count and attempts < count * 3:
            attempts += 1
            traj = generate_trajectory(env, scramble_steps)
            
            if traj and len(traj) >= 5:  # At least 5 moves (filter out trivial puzzles)
                # Store as list of grids (original format for compatibility)
                grids = [t[0] for t in traj]
                all_trajectories.append(np.array(grids))
                success += 1
                generated += 1
                
                if generated % 20 == 0:
                    print(f"  Generated {generated}/{total} trajectories...")
        
        print(f"✓ {num_boxes} boxes, {scramble_steps} scramble: {success} trajectories")
    
    # Save
    dataset = np.array(all_trajectories, dtype=object)
    np.save(output_file, dataset)
    
    print("-" * 50)
    print(f"✅ Saved {len(dataset)} trajectories to '{output_file}'")
    
    # Stats
    total_states = sum(len(t) for t in dataset)
    avg_len = total_states / len(dataset)
    print(f"📊 Total states: {total_states}, Avg trajectory length: {avg_len:.1f}")
    
    return dataset


if __name__ == "__main__":
    generate_dataset()
