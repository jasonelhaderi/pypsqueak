from squal.squalcore import Qubit, Gate, QCSim
import squal.gates

class Program():
    '''
    Program class provides a data structure for composing and organizing programs
    to run on QCSim.
    '''

    def __init__(self):
        self.__instructions = []

    def add_instr(self, gate_target_tuple, position=None):
        '''
        Adds an instruction to self.__instructions, with the default behavior
        being to append the instruction.
        '''

        # Sets default appeding behavior
        if position == None:
            position = len(self)

        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number.')

        if not isinstance(gate_target_tuple, tuple):
            raise TypeError('Argument must be a tuple of Gate object followed by target qubits.')

        elif not isinstance(gate_target_tuple[0], type(Gate())):
            raise TypeError('First element of argument must be a Gate object.')

        for i in range(1, len(gate_target_tuple)):
            if not isinstance(gate_target_tuple[i], int) or gate_target_tuple[i] < 0:
                raise ValueError('Target qubits must be nonnegative integers.')

        self.__instructions.insert(position, gate_target_tuple)

    def rm_instr(self, position=None):
        '''
        Removes an instruction from self.__instructions by index. The default
        behavior is to removed the last instruction.
        '''
        if position == None:
            position = -1

        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number.')

        del self.__instructions[position]

    def execute(self, n_times):
        pass

    def __len__(self):
        return len(self.__instructions)

    def __repr__(self):
        program_rep = ""
        for instruction in self.__instructions:
            program_rep += str(instruction)
            program_rep += '\n'

        program_rep = program_rep.rstrip('\n')
        return str(program_rep)
