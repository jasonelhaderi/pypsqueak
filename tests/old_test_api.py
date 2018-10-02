# Standard modules
import unittest
import numpy as np
import cmath

# pypSQUEAK modules
from pypsqueak.squeakcore import Qubit, Gate
import pypsqueak.gates as gt
import pypsqueak.api as sq
import pypsqueak.errors as sqerr

class qcVirtualMachineSuccess(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

        # Test program
        self.test_program = sq.Program()

    def test_known_measurement_results(self):
        '''Verifies that the proper post-measurement state occurs in several cases'''

        # qcVirtualMachine is initialized in the |0> state, so first let's measure a freshly
        # initialized qcVirtualMachine, and store what should be zero in the 1st classical register
        # location
        p = sq.Program()
        p.measure(0, 1)
        output_q_reg = self.test_qcvm.quantum_state(p)
        output_c_reg = self.test_qcvm.execute(p)
        np.testing.assert_array_equal(output_q_reg.state(), np.array([1, 0]))
        np.testing.assert_array_equal(output_c_reg, [0, 0])

        # Now let's remove that instruction from the program and see that the X
        # gate gives a |1> state by saving the measurement into the 6th classical
        # register location
        p.rm_instr()
        p.add_instr(gt.X(0))
        p.measure(0, 6)
        output_q_reg =self.test_qcvm.quantum_state(p)
        output_c_reg = self.test_qcvm.execute(p)
        np.testing.assert_array_equal(output_q_reg.state(), np.array([0, 1]))
        np.testing.assert_array_equal(output_c_reg, [0, 0, 0, 0, 0, 0, 1])

        # Now let's reset the program to initialize the state |100>, and then measure
        # the 1st and 2nd qubits into classical register locations 3 and 2, respectively,
        # and then measure the 0th qubit without storing the result
        while len(p) > 0:
            p.rm_instr()

        p.add_instr(gt.X(2))
        p.measure(1, 3)
        p.measure(2, 2)
        p.measure(0)
        output_q_reg =self.test_qcvm.quantum_state(p)
        output_c_reg = self.test_qcvm.execute(p)
        np.testing.assert_array_equal(output_q_reg.state(),\
                                      np.array([0, 0, 0, 0, 1, 0, 0, 0]))
        np.testing.assert_array_equal(output_c_reg, [0, 0, 1, 0])

    def test_known_instr_results(self):
        '''
        Verifies the output of several known instructions.
        '''

        test_programs = [sq.Program() for i in range(3)]

        # Takes |0> to |1>
        test_programs[0].add_instr(gt.X(0))

        # Takes |0> to |0001>
        test_programs[1].add_instr(gt.X(0))
        test_programs[1].add_instr(gt.I(3))

        # Takes |0> to (1/sqrt(2))(|000> - |100>)
        test_programs[2].add_instr(gt.I(2))
        test_programs[2].add_instr(gt.X(2))
        test_programs[2].add_instr(gt.H(2))

        results = [np.array([0, 1]),
                   np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   1/np.sqrt(2) * np.array([1, 0, 0, 0, -1, 0, 0, 0])]

        for test_pair in zip(test_programs, results):
            q_reg_output = self.test_qcvm.quantum_state(test_pair[0])
            np.testing.assert_array_almost_equal(q_reg_output.state(), test_pair[1])

    def test_add_distant_qubit(self):
        '''
        A program applying a gate to a non-extant target qubit
        should initialize filler qubits in the |0> state.
        '''

        self.test_program.add_instr(gt.I(2))
        output_q_reg = self.test_qcvm.quantum_state(self.test_program)
        state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(output_q_reg.state(), state_000)

    def test_known_swaps(self):
        '''
        Verifies known swaps in the private self.__swap() method.
        '''
        # We use a hard-coded identity gate to initialize extra qubits
        i_gate = Gate(gt._I)

        # Verify that |001> gets swapped to |100>
        self.test_qcvm._qcVirtualMachine__instr(i_gate, 2)
        self.test_qcvm._qcVirtualMachine__quantum_reg.change_state([0, 1, 0, 0, 0, 0, 0, 0])
        self.test_qcvm._qcVirtualMachine__swap(0, 2)
        state_100 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qcvm._qcVirtualMachine__quantum_reg.state(), state_100)

        self.test_qcvm._qcVirtualMachine__reset()

        # Verify that |100> gets swapped to |010> when qubits 1 and 2 are swapped
        self.test_qcvm._qcVirtualMachine__instr(i_gate, 2)
        self.test_qcvm._qcVirtualMachine__quantum_reg.change_state([0, 0, 0, 0, 1, 0, 0, 0])
        self.test_qcvm._qcVirtualMachine__swap(1, 2)
        state_010 = np.array([0, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qcvm._qcVirtualMachine__quantum_reg.state(), state_010)

        self.test_qcvm._qcVirtualMachine__reset()

        # Verify that (|011> - |010>)/sqrt(2) gets swapped to (|101> - |100>)/sqrt(2)
        # when qubits 1 and 2 are swapped
        self.test_qcvm._qcVirtualMachine__instr(i_gate, 2)
        self.test_qcvm._qcVirtualMachine__quantum_reg.change_state([0, 0, -1, 1, 0, 0, 0, 0])
        self.test_qcvm._qcVirtualMachine__swap(1, 2)
        state_superposition = (1/np.sqrt(2)) * np.array([0, 0, 0, 0, -1, 1, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qcvm._qcVirtualMachine__quantum_reg.state(), state_superposition)

class qcVirtualMachineFailure(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

        # Test program
        self.test_program = sq.Program()

    def test_instr_empty_register(self):
        '''
        The private self.__instr() method must fail when no quantum_reg indicies
        are specified to operate on.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(IndexError, self.test_qcvm._qcVirtualMachine__instr, i_gate)

    def test_instr_negative_loc(self):
        '''
        The private self.__instr() method must fail when specified register
        location is negative.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(ValueError, self.test_qcvm._qcVirtualMachine__instr, i_gate, -1)

    def test_instr_non_int_loc(self):
        '''
        The private self.__instr() method must fail when register location
        isn't integer.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(TypeError, self.test_qcvm._qcVirtualMachine__instr, i_gate, 1.1)

    def test_gate_and_reg_mismatch(self):
        '''
        The private self.__instr() method must fail when the number of qubit
        registers dont match the size of gate.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(sqerr.WrongShapeError, self.test_qcvm._qcVirtualMachine__instr, i_gate, 0, 1)

    def test_duplicate_q_reg_locs(self):
        '''
        The private self.__instr() method must fail when duplicate
        operational register locations are specified for a gate product.
        '''

        i_gate = Gate(gt._I)
        x_gate = Gate(gt._X)
        i_x_gate_product = i_gate.gate_product(x_gate)

        self.assertRaises(ValueError, self.test_qcvm._qcVirtualMachine__instr, i_x_gate_product, 1, 1)

    def test_swap_non_int_input(self):
        '''
        The private self.__swap() method should fail with non-integer input.
        '''

        self.assertRaises(IndexError, self.test_qcvm._qcVirtualMachine__swap, *['peas', []])

    def test_negative_reg_loc(self):
        '''
        The private self.__measure() method must fail with negative registers.
        '''

        self.assertRaises(ValueError, self.test_qcvm._qcVirtualMachine__measure, -1)
        self.assertRaises(ValueError, self.test_qcvm._qcVirtualMachine__measure, 0, -1)

    def test_swap_index_out_of_range(self):
        '''
        The private self.__swap() method must fail if one of the target qubits
        is uninitialized.
        '''

        # Prepare the state |001>
        i_gate = Gate(gt._I)
        x_gate = Gate(gt._X)

        self.test_qcvm._qcVirtualMachine__instr(i_gate, 2)
        self.test_qcvm._qcVirtualMachine__instr(i_gate, 0)

        self.assertRaises(IndexError, self.test_qcvm._qcVirtualMachine__swap, 0, 3)

class ClassicalGateValidInput(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

        # Test program
        self.test_program = sq.Program()

    def test_classical_logic_gates(self):
        '''
        Verifies the action of classical logic gates.
        '''

        # Prepare the classical register state 0, 1, 0
        self.test_program.measure(0, 2)
        self.test_program.add_instr(gt.X(0))
        self.test_program.measure(0, 1)

        # Negate each bit to yield 1, 0, 1
        for i in range(3):
            self.test_program.add_instr(gt.NOT(i))

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [1, 0, 1])

        # Erase program
        while len(self.test_program) > 0:
            self.test_program.rm_instr()

        # Prepare the classical register state 0, 0, 0, 0, 0, 1
        self.test_program.add_instr(gt.X(0))
        self.test_program.measure(0, 5)

        # Use the first for register locations as a truth table for AND
        self.test_program.add_instr(gt.AND(4, 4, 0))
        self.test_program.add_instr(gt.AND(4, 5, 1))
        self.test_program.add_instr(gt.AND(5, 4, 2))
        self.test_program.add_instr(gt.AND(5, 5, 3))

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program),\
                                      [0, 0, 0, 1, 0, 1])

        # Erase program
        while len(self.test_program) > 0:
            self.test_program.rm_instr()

        # Prepare the classical register state 0, 0, 0, 0, 0, 1
        self.test_program.add_instr(gt.X(0))
        self.test_program.measure(0, 5)

        # Use the first for register locations as a truth table for OR
        self.test_program.add_instr(gt.OR(4, 4, 0))
        self.test_program.add_instr(gt.OR(4, 5, 1))
        self.test_program.add_instr(gt.OR(5, 4, 2))
        self.test_program.add_instr(gt.OR(5, 5, 3))

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program),\
                                      [0, 1, 1, 1, 0, 1])

        # Erase program
        while len(self.test_program) > 0:
            self.test_program.rm_instr()

        # Test TRUE and FALSE
        self.test_program.add_instr(gt.TRUE(1))
        self.test_program.add_instr(gt.TRUE(0))
        self.test_program.add_instr(gt.FALSE(1))

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [1, 0])

        # Erase program
        while len(self.test_program) > 0:
            self.test_program.rm_instr()

        # Check COPY and EXCHANGE
        self.test_program.add_instr(gt.TRUE(0))
        for i in range(3):
            self.test_program.add_instr(gt.COPY(0, 2*i))

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [1, 0, 1, 0, 1])

        self.test_program.add_instr(gt.EXCHANGE(4, 3))
        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [1, 0, 1, 1, 0])

class ControlFlowValidInput(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

        # Test program
        self.test_program = sq.Program()

    def test_if_then_else(self):
        '''
        Verifies proper control flow with if/then, and if/then/else statements.
        '''

        # Initialize the classical register to the state [0, 1]
        self.test_program.add_instr(gt.X(0))
        self.test_program.measure(0, 1)

        if_branch = sq.Program()
        else_branch = sq.Program()

        if_branch.add_instr(gt.X(0))
        if_branch.measure(0, 2)

        else_branch.measure(0, 2)

        self.test_program.if_then_else(0, if_branch, else_branch)

        # Should return [0, 1, 1]
        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [0, 1, 1])

        # Now let's use the control bit 1 to see if we return [0, 1, 0]
        self.test_program.rm_instr()
        self.test_program.if_then_else(1, if_branch, else_branch)

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [0, 1, 0])

        # Now let's not specify an else branch, and verify that a 0 as the control
        # bit results in execution passing on

        self.test_program.rm_instr()
        self.test_program.if_then_else(0, if_branch)

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [0, 1])

    def test_while_loop_hadamard(self):
        '''
        Hadamard state measurement of 0 should end appropriately constucted .while_loop().
        '''

        test_register = 0
        save_register = 1
        # First place 1 into the classical test register location
        self.test_program.add_instr(gt.TRUE(test_register))
        # Now initialize a superposition state
        self.test_program.add_instr(gt.H(0))

        # Body of while loop
        body = sq.Program()

        body.measure(0, save_register)
        body.add_instr(gt.COPY(save_register, test_register))
        body.add_instr(gt.NOT(save_register))
        body.add_instr(gt.H(0))
        self.test_program.while_loop(test_register, body)

        np.testing.assert_array_equal(self.test_qcvm.execute(self.test_program), [0, 1])
        np.testing.assert_array_almost_equal(abs(self.test_qcvm.quantum_state(self.test_program).state()),\
                                      np.array([1/np.sqrt(2), 1/np.sqrt(2)]))

class ProgramInvalidInput(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

        # Test program
        self.test_program = sq.Program()

        self.iden = np.eye(2)

    def test_protected_names(self):
        '''
        gate_def and add_instr throw a NameError when the name is a keyword or builtin gate name.
        '''

        builtin_qgates = [key for key in self.test_qcvm._qcVirtualMachine__builtin_qgates]
        builtin_cgates = [key for key in self.test_qcvm._qcVirtualMachine__builtin_cgates]
        keywords = [word for word in sq._keywords]

        for name in builtin_qgates:
            self.assertRaises(NameError, self.test_program.gate_def, name, self.iden)

        for name in builtin_cgates:
            self.assertRaises(NameError, self.test_program.gate_def, name, self.iden)

        for word in keywords:
            self.assertRaises(NameError, self.test_program.gate_def, word, self.iden)
            self.assertRaises(NameError, self.test_program.add_instr, (word, 0))

    def test_first_char(self):
        '''
        gate_def and add_instr throw a NameError when the first char in a name
        is not a letter.
        '''

        bad_names = ['1nae', '_fje', '.blahblah', '#fdsa', '&NAME']

        for name in bad_names:
            self.assertRaises(NameError, self.test_program.gate_def, name, self.iden)
            self.assertRaises(NameError, self.test_program.add_instr, (name, 0))

    def test_empty_name(self):
        '''
        gate_def and add_instr throw a NameError fails when the name is an empty string.
        '''

        self.assertRaises(NameError, self.test_program.gate_def, '', self.iden)
        self.assertRaises(NameError, self.test_program.add_instr, ('', 0))

    def test_malformed_names(self):
        '''
        gate_def and add_instr throw a NameError if name contains protected chars.
        '''

        prefix = 'blah'
        sufix = 'blah'
        bad_names = [prefix + char + sufix for char in sq._protected_chars]

        for name in bad_names:
            self.assertRaises(NameError, self.test_program.gate_def, name, self.iden)
            self.assertRaises(NameError, self.test_program.add_instr, (name, 0))

    def test_invalid_matrix_rep(self):
        '''
        gate_def throws a TypeError if matrix_rep is not a callable, a tuple/list of
        tuples/lists, a numpy array, or if the elements are not numeric.
        '''

        bad_matricies = [{'mydict': 17},
                         [],
                         [(), ()],
                         4,
                         'apples',
                         [[1, 'test'], [5, (2, 4)]],
                         np.array([['train', 4], [12, 45]])]

        for matrix in bad_matricies:
            self.assertRaises(TypeError, self.test_program.gate_def, 'MyGate', matrix)

    def test_bad_gate_target_tuples(self):
        '''
        add_instr throws a TypeError when handed a non-tuple or a malformed gate_target_tuple.
        '''

        not_tuples = [[34],
                      {'key': 'val'},
                      14,
                      'thing']

        bad_tuples = [(['name', 0], 3),
                      (1, 3),
                      ()]

        for candidate in not_tuples:
            self.assertRaises(TypeError, self.test_program.add_instr, candidate)

        for candidate in bad_tuples:
            self.assertRaises(TypeError, self.test_program.add_instr, candidate)

    def test_gate_target_tuple_bad_index(self):
        '''
        add_instr throws an IndexError if specified targets are not nonnegative integers.
        '''

        bad_targets = [('name', -1), ('name', 3, 2, 7j), ('name', 1.1, 2.0, 0)]

        for candidate in bad_targets:
            self.assertRaises(IndexError, self.test_program.add_instr, candidate)

    def test_bad_noise_input(self):
        '''
        add_instr throws a TypeError if kraus_ops isn't a list of matricies.
        '''

        bad_kraus_ops = [[],
                         'nope',
                         {},
                         15,
                         (),
                         np.array([2, 3,]),
                         [[14, 3], 5],
                         [[[45, 2], [14, 3]], 3]]

        for ops in bad_kraus_ops:
            self.assertRaises(TypeError, self.test_program.add_instr, ('name', 0), kraus_ops=ops)

    def test_bad_measurement_input(self):
        '''
        measure raises an IndexError when qubit_loc or an optional classical_loc is not a
        nonnegative integer.
        '''

        bad_locs = [2.4, 8j, -2, 'twelve']

        for loc in bad_locs:
            self.assertRaises(IndexError, self.test_program.measure, loc)
            self.assertRaises(IndexError, self.test_program.measure, 0, loc)

    def test_if_then_else_bad_c_reg(self):
        '''
        if_then_else raises an IndexError when the specified c_reg_test_loc isn't
        a nonnegative integer.
        '''

        bad_locs = [2.4, 8j, -2, 'twelve']

        for loc in bad_locs:
            self.assertRaises(IndexError, self.test_program.if_then_else, loc, sq.Program())

    def test_if_then_else_non_program_branches(self):
        '''
        if_then_else raises a TypeError when given then_ or else_branches that
        aren't Program objects.
        '''

        bad_branches = [[], 15, 'test', {}, ('instruction', 3)]

        for branch in bad_branches:
            self.assertRaises(TypeError, self.test_program.if_then_else, 0, branch)
            self.assertRaises(TypeError, self.test_program.if_then_else, 0, sq.Program(), branch)

    def test_while_loop_bad_c_reg(self):
        '''
        while_loop raises an IndexError when the specified c_reg_test_loc isn't
        a nonnegative integer.
        '''

        bad_locs = [2.4, 8j, -2, 'twelve']

        for loc in bad_locs:
            self.assertRaises(IndexError, self.test_program.while_loop, loc, sq.Program())

    def test_while_loop_non_program_body(self):
        '''
        while_loop raises a TypeError when given a loop_body that isn't a Program object.
        '''

        bad_bodies = [[], 15, 'test', {}, ('instruction', 3)]

        for body in bad_bodies:
            self.assertRaises(TypeError, self.test_program.while_loop, 0, body)

    def test_format_instr_bad_index(self):
        '''
        format_instr raises an IndexError when given an instruction number that
        is not a nonnegative integer or is out of range.
        '''

        self.test_program.add_instr(('some_gate', 3))
        bad_locs = [2.4, 8j, -2, 'twelve', len(self.test_program)]

        for loc in bad_locs:
            self.assertRaises(IndexError, self.test_program.format_instr, loc)

if __name__ == '__main__':
    unittest.main()
