from squal.squalcore import Qubit, Gate, QCSim
import squal.gates

class Program():
    '''
    Program class provides a data structure for composing and organizing programs
    to run on QCSim.
    '''

    def __init__(self):
        self.instructions = []

    def add_instr(self, gate_target_tuple):
        if not isinstance(gate_target_tuple, tuple):
            raise TypeError('Argument must be a tuple of Gate object followed by target qubits.')

        elif not isinstance(gate_target_tuple, type(Gate())):
            raise TypeError('First element of argument must be a Gate object.')

        for i in range(1, len(gate_target_tuple)):
            if not isinstance(gate_target_tuple[i], int):
                raise ValueError('Target qubits must be integer.')

        self.instruction.append(gate_target_tuple)

    def __repr__(self):
        program_rep = ""
        for instruction in self.instructions:
            program_rep += str(instruction)
            program_rep += '\n'
        program_rep = program_rep.rstrip('\n')
        return str(program_rep)
