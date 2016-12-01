# -*- coding: utf-8 -*-
"""
Tests of neo.io.gdfio

Authors: Julia Sprenger, Johanna Senk, Maximilian Schmidt, Jakob Jordan
"""

from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.gdfio import GdfIO
from neo.test.iotest.common_io_test import BaseTestIO
import quantities as pq
import numpy as np


class TestGdfIO(BaseTestIO, unittest.TestCase):
    ioclass = GdfIO
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
        #r = GdfIO(filename='gdf_nest_test_files/withgidF-time_in_stepsF-1254-0.gdf')
        r = GdfIO(filename='gdf_nest_test_files/0time-1255-0.gdf')
        r.read_spiketrain(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                       id_column=None, time_column=0)

        #r = GdfIO(filename='gdf_nest_test_files/withgidF-time_in_stepsT-1256-0.gdf')
        r = GdfIO(filename='gdf_nest_test_files/0time_in_steps-1257-0.gdf')
        r.read_spiketrain(t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column=None, time_column=0)

        #r = GdfIO(filename='gdf_nest_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          lazy=False, id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       lazy=False, id_column=0, time_column=1)

        #r = GdfIO(filename='gdf_nest_test_files/withgidT-time_in_stepsT-1257-0.gdf')
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time_in_steps-1258-0.gdf')
        r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, t_stop=500.*pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_start=400.*pq.ms, t_stop=500.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column=0, time_column=1)


    def test_read_integer(self):
        '''
        Tests if spike times are actually stored as integers if they
        are stored in time steps in the file.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0time_in_steps-1257-0.gdf')
        st = r.read_spiketrain(gdf_id=None, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=None, time_column=0)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gdf_id_list=[None], t_start=400.*pq.ms,
                             t_stop=500.*pq.ms,
                             time_unit=pq.CompoundUnit('0.1*ms'),
                             lazy=False, id_column=None, time_column=0)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))


        r = GdfIO(filename='gdf_nest_test_files/0gid-1time_in_steps-1258-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gdf_id_list=[1], t_start=400.*pq.ms,
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
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms,
                               t_stop=500.*pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.float)
        seg = r.read_segment(gdf_id_list=[1], t_start=400.*pq.ms,
                             t_stop=500.*pq.ms,
                             lazy=False, id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(all([s.magnitude.dtype == np.float for s in sts]))


    def test_values(self):
        '''
        Tests if the routine loads the correct numbers from the file.
        '''
        id_to_test = 1
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        seg = r.read_segment(gdf_id_list=[id_to_test],
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
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        id_list_to_test = range(1,10)
        seg = r.read_segment(gdf_id_list=id_list_to_test, t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gdf_id_list=id_list_to_test, t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == 50)


    def test_read_segment_accepts_range(self):
        '''
        Tests if spiketrains can be retrieved by specifying a range of GDF IDs.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        seg = r.read_segment(gdf_id_list=(10, 39), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == 30)


    def test_read_segment_range_is_reasonable(self):
        '''
        Tests if error is thrown correctly, when second entry is smaller than
        the first one of the range.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        seg = r.read_segment(gdf_id_list=(10, 10), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == 1)
        with self.assertRaises(ValueError):
            seg = r.read_segment(gdf_id_list=(10, 9), t_start=400.*pq.ms,
                                 t_stop=500.*pq.ms, lazy=False,
                                 id_column=0, time_column=1)


    def test_read_spiketrain_annotates(self):
        '''
        Tests if correct annotation is added when reading a spike train.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        ID = 7
        st = r.read_spiketrain(gdf_id=ID, t_start=400.*pq.ms, t_stop=500.*pq.ms)
        self.assertEqual(ID, st.annotations['id'])


    def test_read_segment_annotates(self):
        '''
        Tests if correct annotation is added when reading a segment.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        IDs = (5, 11)
        sts = r.read_segment(gdf_id_list=(5, 11), t_start=400.*pq.ms,
                             t_stop=500.*pq.ms)
        for ID in np.arange(5, 12):
            self.assertEqual(ID, sts.spiketrains[ID-5].annotations['id'])


    def test_adding_custom_annotation(self):
        '''
        Tests if custom annotation is correctly added.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
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
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_segment(t_start=400.*pq.ms, t_stop=500.*pq.ms, lazy=False,
                           id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment()


    def test_t_start_t_stop(self):
        '''
        Tests if the t_start and t_stop arguments are correctly processed.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')

        t_stop_targ = 490.*pq.ms
        t_start_targ = 410.*pq.ms

        seg = r.read_segment(gdf_id_list=[], t_start=t_start_targ,
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
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_stop=500.*pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gdf_id_list=[1, 2, 3], t_stop=500.*pq.ms, lazy=False,
                           id_column=0, time_column=1)


    def test_t_stop_undefined_raises_error(self):
        '''
        Tests if undefined t_stop, i.e., t_stop=None raises error.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_start=400.*pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gdf_id_list=[1, 2, 3], t_start=400.*pq.ms, lazy=False,
                           id_column=0, time_column=1)


    def test_gdf_id_illdefined_raises_error(self):
        '''
        Tests if ill-defined gdf_id in read_spiketrain (i.e., None, list, or
        empty list) raises error.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
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
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        seg = r.read_segment(gdf_id_list=[], t_start=400.*pq.ms, t_stop=1.*pq.ms)
        for st in seg.spiketrains:
            self.assertEqual(st.size, 0)


    def test_read_spiketrain_can_return_empty_spiketrain(self):
        '''
        Tests if read_spiketrain returns an empty SpikeTrain if no spikes are in
        time range.
        '''
        r = GdfIO(filename='gdf_nest_test_files/0gid-1time-1256-0.gdf')
        st = r.read_spiketrain(gdf_id=0, t_start=400.*pq.ms, t_stop=1.*pq.ms)
        self.assertEqual(st.size, 0)


if __name__ == "__main__":
    unittest.main()
