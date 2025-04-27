class TuringMachine:
    def __init__(self, num_states=3, num_symbols=2):
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.transitions = {}
        self.tape = {}
        self.head = 0
        self.current_state = 0
        self.halted = False

    def add_transition(self, state, symbol, new_symbol, direction, new_state):
        self.transitions[(state, symbol)] = (new_symbol, direction, new_state)

    def step(self):
        if self.halted:
            return
        current_symbol = self.tape.get(self.head, 0)
        key = (self.current_state, current_symbol)
        if key not in self.transitions:
            self.halted = True
            return
        new_symbol, direction, new_state = self.transitions[key]
        if new_symbol == 0 and self.head in self.tape:
            del self.tape[self.head]
        else:
            self.tape[self.head] = new_symbol
        self.head += 1 if direction == 'R' else -1
        self.current_state = new_state

    def run(self, max_steps=10000, visualize=False):
        steps = 0
        while not self.halted and steps < max_steps:
            if visualize:
                self.visualize()
            self.step()
            steps += 1
        if visualize:
            self.visualize()
        return steps

    def reset(self):
        self.tape.clear()
        self.head = 0
        self.current_state = 0
        self.halted = False

    def serialize(self):
        arr = []
        for state in range(self.num_states):
            for symbol in range(self.num_symbols):
                key = (state, symbol)
                if key in self.transitions:
                    new_symbol, direction, new_state = self.transitions[key]
                    dir_bit = 0 if direction == 'L' else 1
                    arr.append((new_symbol, dir_bit, new_state))
                else:
                    arr.append((-1, 0, -1))
        return arr

    def visualize(self, window=10):
        """Display a small window around the head."""
        tape_keys = sorted(self.tape.keys())
        if not tape_keys:
            tape_range = range(self.head - window, self.head + window + 1)
        else:
            min_pos = min(min(tape_keys), self.head) - window
            max_pos = max(max(tape_keys), self.head) + window
            tape_range = range(min_pos, max_pos + 1)

        tape_str = ""
        head_str = ""
        for pos in tape_range:
            symbol = self.tape.get(pos, 0)
            tape_str += f"{symbol} "
            head_str += "^ " if pos == self.head else "  "
        print(tape_str.strip())
        print(head_str.strip())

        # Also show rules below
        print(f"State: {self.current_state}, Halted: {self.halted}")
