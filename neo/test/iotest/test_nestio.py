# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division
import warnings
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq
import numpy as np

from neo.io.nestio import ColumnIO
from neo.io.nestio import NestIO
from neo.test.iotest.common_io_test import BaseTestIO



class TestNestIO(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = []

    def test_read_analogsignalarray(self):
        '''
        Tests reading files in the 4 different formats:
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        '''

        r = NestIO(filenames=
                   'gdf_nest_test_files/0gid-1time-2gex-3Vm-1261-0.dat',
                   # 'nest_test_files/withgidT-time_in_stepsF-1259-0.dat'
                   )
        r.read_analogsignalarray(gid=1, t_stop=1000.*pq.ms,
                                 sampling_period=pq.ms, lazy=False,
                                 id_column=0, time_column=1,
                                 value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                       sampling_period=pq.ms, lazy=False, id_column=0,
                       time_column=1, value_columns=2,
                       value_types='V_m')

        r = NestIO(filenames=
                   'gdf_nest_test_files/0gid-1time-2Vm-3Iex-4Iin-1263-0.dat',
                   # 'nest_test_files/withgidT-time_in_stepsT-1261-0.dat'
                   )
        r.read_analogsignalarray(gid=1, t_stop=1000.*pq.ms,
                                 time_unit=pq.CompoundUnit('0.1*ms'),
                                 sampling_period=pq.ms, lazy=False,
                                 id_column=0, time_column=1,
                                 value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column=0,
                       time_column=1, value_columns=2,
                       value_types='V_m')

    def test_id_column_none_multiple_neurons(self):
        '''
        Tests if function correctly raises an error if the user tries to read
        from a file which does not contain neuron IDs, but data for multiple
        neurons.
        '''

        r = NestIO(filenames=
                   'gdf_nest_test_files/0time-1255-0.gdf',
                   # 'nest_test_files/withgidF-time_in_stepsF-1258-0.dat'
                   )
        with self.assertRaises(ValueError):
            r.read_analogsignalarray(t_stop=1000.*pq.ms, lazy=False,
                                     sampling_period=pq.ms,
                                     id_column=None, time_column=0,
                                     value_column=1)
            r.read_segment(t_stop=1000.*pq.ms, lazy=False,
                           sampling_period=pq.ms, id_column=None,
                           time_column=0, value_column=1)

        # r = NestIO(filenames='nest_test_files/withgidF-time_in_stepsT-1260-0'
        #                      '.dat')
        # with self.assertRaises(ValueError):
        #     r.read_analogsignalarray(t_stop=1000.*pq.ms,
        #                              time_unit=pq.CompoundUnit('0.1*ms'),
        #                              lazy=False, id_column=None,
        #                              time_column=0, value_column=1)
        #     r.read_segment(t_stop=1000.*pq.ms,
        #                    time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
        #                    id_column=None, time_column=0, value_column=1)



    def test_values(self):
        '''
        Tests if the function returns the correct values.
        '''

        filename = 'gdf_nest_test_files/0gid-1time-2gex-3Vm-1261-0.dat'

        id_to_test = 1
        r = NestIO(filenames=filename)
        seg = r.read_segment(gid_list=[id_to_test],
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_columns=2, value_types='V_m')

        dat = np.loadtxt(filename)
        target_data = dat[:, 2][np.where(dat[:, 0] == id_to_test)]
        st = seg.analogsignalarrays[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        '''
        Tests if signals are correctly stored in a segment.
        '''

        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-2gex-1262-0.dat')

        id_list_to_test = range(1, 10)
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_columns=2, value_types='V_m')

        self.assertTrue(len(seg.analogsignalarrays) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_columns=2, value_types='V_m')

        self.assertEqual(len(seg.analogsignalarrays), 50)

    def test_wrong_input(self):
        '''
        Tests two cases of wrong user input, namely
        - User does not specify a value column
        - User does not make any specifications
        - User does not define sampling_period as a unit
        - User specifies a non-default value type without
          specifying a value_unit
        - User specifies t_start < 1.*sampling_period
        '''

        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-2gex-1262-0.dat')
        with self.assertRaises(ValueError):
            r.read_segment(t_stop=1000.*pq.ms, lazy=False,
                           id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment()
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                           sampling_period=1.*pq.ms, lazy=False,
                           id_column=0, time_column=1,
                           value_columns=2, value_types='V_m')

        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                           sampling_period=pq.ms, lazy=False,
                           id_column=0, time_column=1,
                           value_columns=2, value_types='U_mem')

        # with self.assertRaises(ValueError):
        #     r.read_segment(gid_list=[1], t_start=0.*pq.ms, t_stop=1000.*pq.ms,
        #                    sampling_period=pq.ms, lazy=False,
        #                    id_column=0, time_column=1,
        #                    value_columns=2, value_types='V_m')


    def test_t_start_t_stop(self):
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-2gex-1262-0.dat')

        t_start_targ = 450.*pq.ms
        t_stop_targ = 480.*pq.ms

        seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column=0, time_column=1,
                             value_columns=2, value_types='V_m')
        sts = seg.analogsignalarrays
        for st in sts:
            self.assertTrue(st.t_start == t_start_targ)
            self.assertTrue(st.t_stop == t_stop_targ)


class TestColumnIO(unittest.TestCase):

    def setUp(self):
        filename = 'gdf_nest_test_files/0gid-1time-2Vm-3gex-4gin-1260-0.dat'
        self.testIO = ColumnIO(filename=filename)

    def test_no_arguments(self):
        columns = self.testIO.get_columns()
        expected = self.testIO.data
        np.testing.assert_array_equal(columns,expected)

    def test_single_column_id(self):
        column = self.testIO.get_columns(column_ids=1)
        expected = self.testIO.data[:,1]
        np.testing.assert_array_equal(column,expected)

    def test_multiple_column_ids(self):
        columns = self.testIO.get_columns(column_ids=range(2))
        expected = self.testIO.data[:,[0,1]]
        np.testing.assert_array_equal(columns,expected)

    def test_no_condition(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            self.testIO.get_columns(condition_column=0)
            # Verify number and content of warning
            assert len(w) == 1
            assert "no condition" in str(w[-1].message)

    def test_no_condition_column(self):
        with self.assertRaises(ValueError) as context:
            self.testIO.get_columns(condition=lambda x: True)

        self.assertTrue('no condition_column ID provided' in
                        context.exception.message)

    def test_correct_condition_selection(self):
        condition_column = 0
        condition_function = lambda x:x>10
        result = self.testIO.get_columns(condition=condition_function,
                                         condition_column=0)
        selected_ids = np.where(condition_function(self.testIO.data[:,
                                                   condition_column]))[0]
        expected = self.testIO.data[selected_ids,:]

        np.testing.assert_array_equal(result,expected)

        assert all(condition_function(result[:,condition_column]))

    def test_sorting(self):
        result = self.testIO.get_columns(sorting_columns=0)

        assert len(result) > 0
        assert all(np.diff(result[:,0])>=0)



if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestNestIO('test_read_analogsignalarray'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
