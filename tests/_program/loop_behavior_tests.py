import unittest
import numpy as np

from qctoolkit.pulses import TablePT, RepetitionPT, ForLoopPT, SequencePT
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit._program._loop import MultiChannelProgram
from qctoolkit._program.instructions import REPJInstruction, EXECInstruction


class LoopMeasurementsBehaviorTests(unittest.TestCase):

    def test_fpt_rpt_tpt_with_measurements(self) -> None:
        tpt_C = TablePT({0: [(0, 0), (1, 2.5), (2, 'v', 'linear')]}, measurements=[('c', 1, 1)])
        rpt_B = RepetitionPT(tpt_C, 'k', measurements=[('b', 1, 4)])
        fpt_A = ForLoopPT(rpt_B, 'v', (4, 6), measurements=[('a', 2, 4)])
        parameters = {'k': 3}
        sequencer = Sequencer()
        sequencer.push(fpt_A, parameters)
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        # tpt_C has duration 2, rpt_B with k=3 has duration 6, fpt_A has duration 12

        program = MultiChannelProgram(instruction_block)

        expected_measurements = {'a': ([2.], [4.]),
                                 'b': ([1., 7.], [4., 4.]),
                                 'c': ([1., 3., 5., 7., 9., 11.], [1., 1., 1., 1., 1., 1.])
                                 }
        measurements = program.programs[frozenset({0})].get_measurement_windows()
        self.assertEqual(expected_measurements.keys(), measurements.keys())
        for k in expected_measurements:
            self.assertTrue(np.array_equal(expected_measurements[k][0], measurements[k][0]))
            self.assertTrue(np.array_equal(expected_measurements[k][1], measurements[k][1]))

    def test_rpt_rpt_tpt_with_measurements(self) -> None:
        tpt_C = TablePT({0: [(0, 0), (1, 2.5), (2, 'v', 'linear')]}, measurements=[('c', 1, 1)])
        rpt_B = RepetitionPT(tpt_C, 'k', measurements=[('b', 1, 4)])
        rpt_A = RepetitionPT(rpt_B, 'v', measurements=[('a', 2, 4)])
        parameters = {'k': 3, 'v': 2}
        sequencer = Sequencer()
        sequencer.push(rpt_A, parameters)
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        # tpt_C has duration 2, rpt_B with k=3 has duration 6, fpt_A with v=2 has duration 12

        program = MultiChannelProgram(instruction_block)

        expected_measurements = {'a': ([2.], [4.]),
                                 'b': ([1., 7.], [4., 4.]),
                                 'c': ([1., 3., 5., 7., 9., 11.], [1., 1., 1., 1., 1., 1.])
                                 }
        measurements = program.programs[frozenset({0})].get_measurement_windows()
        self.assertEqual(expected_measurements.keys(), measurements.keys())
        for k in expected_measurements:
            self.assertTrue(np.array_equal(expected_measurements[k][0], measurements[k][0]))
            self.assertTrue(np.array_equal(expected_measurements[k][1], measurements[k][1]))

    def test_tpt_rpt_tpt_with_measurement_tpt_duration_zero(self) -> None:
        tpt_C = TablePT({0: [('zero', 'v')]}, measurements=[('c', 1, 1)])
        rpt_B = RepetitionPT(tpt_C, 'k', measurements=[('b', 1, 4)])
        fpt_A = ForLoopPT(rpt_B, 'v', (4, 6), measurements=[('a', 2, 4)])
        parameters = {'k': 3, 'zero': 0}
        sequencer = Sequencer()
        sequencer.push(fpt_A, parameters)
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        # length of tpt_C is zero so no EXEC instruction will be generated in rpt_B REPJ block. assert!
        self.assertIsInstance(instruction_block.instructions[2], REPJInstruction)
        repj_inst = instruction_block.instructions[2] # type: REPJInstruction
        self.assertEqual(0, len(repj_inst.target.block.instructions))

        # tpt_C has duration 0, rpt_B with k=3 has duration 0, fpt_A has duration 0

        # no waveforms in instruction block => no channels defined => error
        with self.assertRaises(ValueError):
            MultiChannelProgram(instruction_block)

    def test_tpt_rpt_spt_2tpt_with_measurement_one_tpt_duration_zero(self) -> None:
        tpt_C1 = TablePT({0: [('zero', 'v')]}, measurements=[('c', 1, 1)])
        tpt_C2 = TablePT({0: [(0, 1), (1, 0)]})
        spt_C = SequencePT(tpt_C1, tpt_C2)
        rpt_B = RepetitionPT(spt_C, 'k', measurements=[('b', 1, 4)])
        fpt_A = ForLoopPT(rpt_B, 'v', (4, 6), measurements=[('a', 2, 4)])
        parameters = {'k': 3, 'zero': 0}
        sequencer = Sequencer()
        sequencer.push(fpt_A, parameters)
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        # length of tpt_C1 is zero so only one (not two) EXEC instruction will be generated in rpt_B REPJ block. assert!
        self.assertIsInstance(instruction_block.instructions[2], REPJInstruction)
        repj_inst = instruction_block.instructions[2]  # type: REPJInstruction
        self.assertEqual(1, len([i for i in repj_inst.target.block.instructions if isinstance(i, EXECInstruction)]))

        # spt_C has duration 1, rpt_B with k=3 has duration 3, fpt_A has duration 6

        program = MultiChannelProgram(instruction_block)

        # since duration of tpt_C1 is 0, measurement c is dropped
        expected_measurements = {'a': ([2.], [4.]),
                                 'b': ([1., 4.], [4., 4.])
                                 }
        measurements = program.programs[frozenset({0})].get_measurement_windows()
        self.assertEqual(expected_measurements.keys(), measurements.keys())
        for k in expected_measurements:
            self.assertTrue(np.array_equal(expected_measurements[k][0], measurements[k][0]))
            self.assertTrue(np.array_equal(expected_measurements[k][1], measurements[k][1]))

    def test_rpt_spt_with_measurement_duration_zero(self) -> None:
        tpt1 = TablePT({0: [(0, 0), (1, 1)]})
        tpt2 = TablePT({0: [('zero', 7)]})
        spt_in = SequencePT(tpt2, tpt2, measurements=[('s_in', 1, 1)])
        spt_out = SequencePT(tpt1, spt_in, measurements=[('s_out', 1, 1)])
        rpt = RepetitionPT(spt_out, 2)

        sequencer = Sequencer()
        sequencer.push(rpt, parameters={'zero': 0})
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        program = MultiChannelProgram(instruction_block)
        # hehavior in current Sequencer: s_in has no duration but still define measurements (in comparison: an AtomicPT with no duration does not define measurements)
        # todo (2018-08-02): define consistent behavior for whether pulse templates define measurements when they have zero duration or not
        expected_measurements = {'s_out': ([1., 2.], [1., 1.]),
                                 's_in': ([2., 3.], [1., 1.])}
        measurements = program.programs[frozenset({0})].get_measurement_windows()
        self.assertEqual(expected_measurements.keys(), measurements.keys())
        for k in expected_measurements:
            self.assertTrue(np.array_equal(expected_measurements[k][0], measurements[k][0]))
            self.assertTrue(np.array_equal(expected_measurements[k][1], measurements[k][1]))

    def test_rpt_spt_with_measurement(self) -> None:
        tpt1 = TablePT({0: [(0, 0), (1, 1)]})
        tpt2 = TablePT({0: [(0, 7), (1, 0, 'linear')]})
        spt_in = SequencePT(tpt2, tpt2, measurements=[('s_in', 1, 2)])
        spt_out = SequencePT(tpt1, spt_in, measurements=[('s_out', 1, 1)])
        rpt = RepetitionPT(spt_out, 2)

        self.assertEqual(2, spt_in.duration.evaluate_numeric())
        self.assertEqual(3, spt_out.duration.evaluate_numeric())

        sequencer = Sequencer()
        sequencer.push(rpt)
        instruction_block = sequencer.build()
        self.assertTrue(sequencer.has_finished())

        program = MultiChannelProgram(instruction_block)
        expected_measurements = {'s_out': ([1., 4.], [1., 1.]),
                                 's_in': ([2., 5.], [2., 2.])}
        measurements = program.programs[frozenset({0})].get_measurement_windows()
        self.assertEqual(expected_measurements.keys(), measurements.keys())
        for k in expected_measurements:
            self.assertTrue(np.array_equal(expected_measurements[k][0], measurements[k][0]))
            self.assertTrue(np.array_equal(expected_measurements[k][1], measurements[k][1]))
