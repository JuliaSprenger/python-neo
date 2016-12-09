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



class TestNestIO_Analogsignals(BaseTestIO, unittest.TestCase):
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


    def test_notimeid(self):
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-2gex-1262-0.dat')

        t_start_targ = 450.*pq.ms
        t_stop_targ = 460.*pq.ms
        sampling_period = pq.CompoundUnit('5*ms')

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                                 sampling_period=sampling_period,
                                 t_stop=t_stop_targ, lazy=False,
                                 id_column=0, time_column=None,
                                 value_columns=2, value_types='V_m')
            # Verify number and content of warning
            self.assertEqual(len(w),1)
            self.assertIn ("no time column id",str(w[0].message))
        sts = seg.analogsignalarrays
        for st in sts:
            self.assertTrue(st.t_start == 1*5*pq.ms)
            self.assertTrue(st.t_stop == len(st)*sampling_period + 1*5*pq.ms)


# class TestNestIOSegment(BaseTestIO, unittest.TestCase):

class TestNestIO_Spiketrains(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = []


    def test_read_spiketrain(self):
        '''
        Tests reading files in the 4 different formats:
        - without GIDs, with times as floats
        - without GIDs, with times as integers in time steps
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        '''
        #r = NestIO(filenames='gdf_nest_test_files/withgidF-time_in_stepsF-1254-0.gdf')
        r = NestIO(filenames='gdf_nest_test_files/0time-1255-0.gdf')
        r.read_spiketrain(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                       id_column=None, time_column=0)

        #r = NestIO(filenames='gdf_nest_test_files/withgidF-time_in_stepsT-1256-0.gdf')
        r = NestIO(filenames='gdf_nest_test_files/0time_in_steps-1257-0.gdf')
        r.read_spiketrain(t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column=None, time_column=0)

        #r = NestIO(filenames='gdf_nest_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          lazy=False, id_column=0, time_column=1)
        r.read_segment(gid_list=[1], t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       lazy=False, id_column=0, time_column=1)

        #r = NestIO(filenames='gdf_nest_test_files/withgidT-time_in_stepsT-1257-0.gdf')
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time_in_steps-1258-0.gdf')
        r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=0, time_column=1)
        r.read_segment(gid_list=[1], t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column=0, time_column=1)


    def test_read_integer(self):
        '''
        Tests if spike times are actually stored as integers if they
        are stored in time steps in the file.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0time_in_steps-1257-0.gdf')
        st = r.read_spiketrain(gdf_id=None, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=None, time_column=0)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gid_list=[None], t_start=400.*pq.ms,
                             t_stop=500.*pq.ms,
                             time_unit=pq.CompoundUnit('0.1*ms'),
                             lazy=False, id_column=None, time_column=0)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))


        r = NestIO(filenames='gdf_nest_test_files/0gid-1time_in_steps-1258-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gid_list=[1], t_start=400.*pq.ms,
                             t_stop=500.*pq.ms,
                             time_unit=pq.CompoundUnit('0.1*ms'),
                             lazy=False, id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))


    def test_read_float(self):
        '''
        Tests if spike times are stored as floats if they
        are stored as floats in the file.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.float)
        seg = r.read_segment(gid_list=[1], t_start=400.*pq.ms,
                             t_stop=500.*pq.ms,
                             lazy=False, id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(all([s.magnitude.dtype == np.float for s in sts]))


    def test_values(self):
        '''
        Tests if the routine loads the correct numbers from the file.
        '''
        id_to_test = 1
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        seg = r.read_segment(gid_list=[id_to_test],
                             t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)

        dat = np.loadtxt('gdf_nest_test_files/0gid-1time-1256-0.gdf')
        target_data = dat[:, 1][np.where(dat[:, 0]==id_to_test)]

        st = seg.spiketrains[0]
        np.testing.assert_array_equal(st.magnitude, target_data)


    def test_read_segment(self):
        '''
        Tests if spiketrains are correctly stored in a segment.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        id_list_to_test = range(1,10)
        seg = r.read_segment(gid_list=id_list_to_test, t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gid_list=id_list_to_test, t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == 50)


    def test_read_segment_accepts_range(self):
        '''
        Tests if spiketrains can be retrieved by specifying a range of GDF IDs.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        seg = r.read_segment(gid_list=(10, 39), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertEqual(len(seg.spiketrains), 30)


    def test_read_segment_range_is_reasonable(self):
        '''
        Tests if error is thrown correctly, when second entry is smaller than
        the first one of the range.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        seg = r.read_segment(gid_list=(10, 10), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertEqual(len(seg.spiketrains), 1)
        with self.assertRaises(ValueError):
            seg = r.read_segment(gid_list=(10, 9), t_start=400.*pq.ms,
                                 t_stop=500.*pq.ms, lazy=False,
                                 id_column=0, time_column=1)


    def test_read_spiketrain_annotates(self):
        '''
        Tests if correct annotation is added when reading a spike train.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        ID = 7
        st = r.read_spiketrain(gdf_id=ID, t_start=400.*pq.ms, t_stop=500.*pq.ms)
        self.assertEqual(ID, st.annotations['id'])


    def test_read_segment_annotates(self):
        '''
        Tests if correct annotation is added when reading a segment.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        IDs = (5, 11)
        sts = r.read_segment(gid_list=(5, 11), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms)
        for ID in np.arange(5, 12):
            self.assertEqual(ID, sts.spiketrains[ID-5].annotations['id'])


    def test_adding_custom_annotation(self):
        '''
        Tests if custom annotation is correctly added.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        st = r.read_spiketrain(gdf_id=0, t_start=400.*pq.ms, t_stop=500.*pq.ms,
                               layer='L23', population='I')
        self.assertEqual(0, st.annotations.pop('id'))
        self.assertEqual('L23', st.annotations.pop('layer'))
        self.assertEqual('I', st.annotations.pop('population'))
        self.assertEqual({}, st.annotations)


    def test_wrong_input(self):
        '''
        Tests two cases of wrong user input, namely
        - User does not specify neuron IDs although the file contains IDs.
        - User does not make any specifications.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                           id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment()


    def test_t_start_t_stop(self):
        '''
        Tests if the t_start and t_stop arguments are correctly processed.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        t_stop_targ = 490.*pq.ms
        t_start_targ = 410.*pq.ms

        seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(np.max([np.max(st.magnitude) for st in sts]) <
                        t_stop_targ.rescale(sts[0].times.units).magnitude)
        self.assertTrue(np.min([np.min(st.magnitude) for st in sts])
                        >= t_start_targ.rescale(sts[0].times.units).magnitude)


    def test_t_start_undefined_raises_error(self):
        '''
        Tests if undefined t_start, i.e., t_start=None raises error.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_stop=500.*pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1, 2, 3], t_stop=500.*pq.ms, lazy=False,
                           id_column=0, time_column=1)


    def test_t_stop_undefined_raises_error(self):
        '''
        Tests if undefined t_stop, i.e., t_stop=None raises error.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1, 2, 3], t_start=400.*pq.ms, lazy=False,
                           id_column=0, time_column=1)


    def test_gdf_id_illdefined_raises_error(self):
        '''
        Tests if ill-defined gdf_id in read_spiketrain (i.e., None, list, or
        empty list) raises error.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=[], t_start=400.*pq.ms, t_stop=500.*pq.ms)
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=[1], t_start=400.*pq.ms, t_stop=500.*pq.ms)
        with self.assertRaises(ValueError):
            r.read_spiketrain(t_start=400.*pq.ms, t_stop=500.*pq.ms)


    def test_read_segment_can_return_empty_spiketrains(self):
        '''
        Tests if read_segment makes sure that only non-zero spike trains are
        returned.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        seg = r.read_segment(gid_list=[], t_start=400.*pq.ms, t_stop=1.*pq.ms)
        for st in seg.spiketrains:
            self.assertEqual(st.size, 0)


    def test_read_spiketrain_can_return_empty_spiketrain(self):
        '''
        Tests if read_spiketrain returns an empty SpikeTrain if no spikes are in
        time range.
        '''
        r = NestIO(filenames='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        st = r.read_spiketrain(gdf_id=0, t_start=400.*pq.ms, t_stop=1.*pq.ms)
        self.assertEqual(st.size, 0)



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
