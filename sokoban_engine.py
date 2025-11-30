import numpy as np
import random

# --- Constants ---
# 0: Floor, 1: Wall, 2: Player, 3: Box, 4: Target, 5: Box on Target
FLOOR, WALL, PLAYER, BOX, TARGET, BOX_TARGET = 0, 1, 2, 3, 4, 5

# Actions dictionary mapping names to (row_change, col_change)
ACTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

class SokobanEnv:
    def __init__(self, width=8, height=8, num_boxes=2):
        self.width = width
        self.height = height
        self.num_boxes = num_boxes
        
        # Two grids: one for dynamic objects (Player, Box), one for static logic (Targets)
        self.grid = np.zeros((height, width), dtype=int)
        self.targets = np.zeros((height, width), dtype=bool) 
        
        self.player_pos = (0, 0)
        
        # Start in a solved state by default (useful for reverse generation)
        self.reset_solved()

    def reset_solved(self):
        """Initializes a perfectly solved state (Boxes on Targets)."""
        self.grid.fill(FLOOR)
        self.targets.fill(False)
        
        # 1. Build Walls around the edges
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL
        
        # 2. Place Player in a random empty spot
        self.player_pos = self._get_random_free_pos()
        self.grid[self.player_pos] = PLAYER
        
        # 3. Place Boxes directly ON Targets
        for _ in range(self.num_boxes):
            r, c = self._get_random_free_pos()
            self.grid[r, c] = BOX_TARGET
            self.targets[r, c] = True # Mark this floor tile as a target permanently
            
        return self.grid.copy()

    def _get_random_free_pos(self):
        """Finds a random coordinate that isn't a wall, player, or existing target."""
        while True:
            r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
            # Ensure we don't spawn on top of something occupied
            if self.grid[r, c] == FLOOR and not self.targets[r, c]:
                return (r, c)

    def step_reverse(self):
        """
        Helper for Scrambling: Performs one random 'Reverse' (Pull) move.
        Used by the solver to generate the problem state.
        Prioritizes pulling boxes off targets.
        """
        # First, try to find a box on a target and pull it off
        boxes_on_targets = np.argwhere(self.grid == BOX_TARGET)
        if len(boxes_on_targets) > 0:
            # Try to pull a box off a target (higher priority)
            # Shuffle directions to try different approaches
            directions = list(ACTIONS.keys())
            random.shuffle(directions)
            
            for action_name in directions:
                dr, dc = ACTIONS[action_name]
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
                    return
        
        # Otherwise, do a normal random reverse move
        action_name = random.choice(list(ACTIONS.keys()))
        self.step(action_name, mode='reverse')

    def step(self, action_name, mode='play'):
        """
        Executes a move in the environment.
        
        Args:
            action_name: 'UP', 'DOWN', 'LEFT', 'RIGHT'
            mode: 
                'play' -> Standard Sokoban rules (Push boxes).
                'reverse' -> Reverse rules (Pull boxes) for scrambling.
        """
        dr, dc = ACTIONS[action_name]
        pr, pc = self.player_pos       # Current Player Position
        nr, nc = pr + dr, pc + dc      # New Player Position
        nnr, nnc = nr + dr, nc + dc    # Two steps ahead (for push check)

        # --- PLAY MODE (Standard Game) ---
        if mode == 'play':
            # 1. Check Bounds/Walls
            if self.grid[nr, nc] == WALL:
                return self.grid.copy()
            
            # 2. Move into Empty Space (Floor or Empty Target)
            if self.grid[nr, nc] in [FLOOR, TARGET]:
                self._move_entity(pr, pc, nr, nc, is_player=True)
                return self.grid.copy()

            # 3. Push Box
            if self.grid[nr, nc] in [BOX, BOX_TARGET]:
                # Can we push it? Is the space *behind* the box empty?
                if self.grid[nnr, nnc] in [FLOOR, TARGET]:
                    # Move Box first: nr,nc -> nnr,nnc
                    self._move_entity(nr, nc, nnr, nnc, is_player=False)
                    # Then Move Player: pr,pc -> nr,nc
                    self._move_entity(pr, pc, nr, nc, is_player=True)
                    return self.grid.copy()

        # --- REVERSE MODE (Scrambling Logic) ---
        elif mode == 'reverse':
            # Check bounds
            if not (0 < nr < self.height-1 and 0 < nc < self.width-1): return self.grid.copy()
            if self.grid[nr, nc] == WALL: return self.grid.copy()
            
            # In reverse, we can't walk BACKWARDS into a box 
            # (unless we assume we are un-pushing it, but let's keep it simple: strict movement)
            if self.grid[nr, nc] in [BOX, BOX_TARGET]: return self.grid.copy()

            # Check for Pull Opportunity:
            # If we move P -> NewP, is there a box BEHIND P (at pr-dr, pc-dc)?
            br, bc = pr - dr, pc - dc 
            should_pull = False
            
            # Verify bounds for the "Behind" spot
            if 0 < br < self.height-1 and 0 < bc < self.width-1:
                # If there is a box behind us, we MIGHT pull it
                if self.grid[br, bc] in [BOX, BOX_TARGET]:
                    # Higher chance to pull boxes off targets (90%), regular boxes (70%)
                    if self.grid[br, bc] == BOX_TARGET:
                        should_pull = (random.random() > 0.1)  # 90% chance
                    else:
                        should_pull = (random.random() > 0.3)  # 70% chance

            if should_pull:
                # 1. Move Player: P -> NewP
                self._move_entity(pr, pc, nr, nc, is_player=True)
                # 2. Pull Box: BehindP -> P (The spot player just left)
                self._move_entity(br, bc, pr, pc, is_player=False)
            else:
                # Just Walk
                self._move_entity(pr, pc, nr, nc, is_player=True)

        return self.grid.copy()

    def _move_entity(self, r, c, nr, nc, is_player):
        """Handles the low-level grid updates and preserves Target memory."""
        # 1. Handle the tile we are LEAVING
        # If it was a target in memory, revert to TARGET visual, else FLOOR
        was_target = self.targets[r, c]
        self.grid[r, c] = TARGET if was_target else FLOOR
        
        # 2. Handle the tile we are ENTERING
        is_target_dest = self.targets[nr, nc]
        
        if is_player:
            self.grid[nr, nc] = PLAYER
            self.player_pos = (nr, nc)
        else:
            # If it's a box moving onto a target, it becomes BOX_TARGET (5)
            self.grid[nr, nc] = BOX_TARGET if is_target_dest else BOX
