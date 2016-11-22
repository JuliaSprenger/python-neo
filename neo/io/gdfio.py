# -*- coding: utf-8 -*-
"""
Class for reading GDF data files, e.g., the spike output of NEST.

Depends on: numpy, quantities

Supported: Read

Usage:
One can read just a single spiketrain with a certain ID from a file
by using read_spiketrain.
    >>> from neo.io import gdfio
    >>> r = gdfio.GdfIO(filename='example_data.gdf')
    >>> st = r.read_spiketrain(gdf_id=8, t_start=0.*pq.ms, t_stop=1000.*pq.ms)
    >>> print(st.magnitude)
    [ 370.6  764.7]
    >>> print(st.annotations)
    {'id': 8}

It is also possible to provide additional annotations for the spiketrain
upon loading the spiketrain
    >>> st = r.read_spiketrain(gdf_id=1, t_start=0.*pq.ms, t_stop=1000.*pq.ms,
                               layer='L6', population='E')
    >>> print(st.annotations)
    {'layer': 'L6', 'id': 1, 'population': 'E'}

One can read multiple spiketrain from a file by passing a list of
IDs to read_segment
    >>> st = r.read_segment(gdf_id_list=[1,6,8], t_start=0.*pq.ms,
                            t_stop=1000.*pq.ms)
    >>> print(st.spiketrains)
    [<SpikeTrain(array([ 354. ,  603.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 274.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 370.6, 764.7]) * ms, [0.0 ms, 1000.0 ms])>]


It is also possible to retrieve spiketrains from a file for all neurons
with at least one spike in the defined period
    >>> st = r.read_segment(gdf_id_list=[], t_start=0.*pq.ms,
                            t_stop=1000.*pq.ms)
    [<SpikeTrain(array([ 411.]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 354. ,  603.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 691.7]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 274.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 370.6,  764.7]) * ms, [0.0 ms, 1000.0 ms])>]


Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk, Jakob Jordan

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, SpikeTrain


class GdfIO(BaseIO):

    """
    Class for reading GDF files, e.g., the spike output of NEST. It handles
    opening the gdf file and reading segements or single spiketrains.
    """

    # This class can only read data
    is_readable = True
    is_writable = False

    supported_objects = [SpikeTrain]
    readable_objects = [SpikeTrain]

    has_header = False
    is_streameable = False

    name = 'gdf'
    extensions = ['gdf']
    mode = 'file'

    def __init__(self, filename=None):
        """
        Parameters
        ----------
            filename: string, default=None
                The filename.
        """
        BaseIO.__init__(self, filename=filename)

    def __read_spiketrains(self, gdf_id_list, time_unit,
                           t_start, t_stop, id_column,
                           time_column, **args):
        """
        Internal function called by read_spiketrain() and read_segment().
        """

        # assert that there are spike times in the file
        if time_column is None:
            raise ValueError('Time column is None. No spike times to '
                             'be read in.')

        if None in gdf_id_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column ' + str(id_column) + '.'
                             ' Specify empty list to retrieve'
                             ' spiketrains of all neurons.')

        if gdf_id_list != [None] and id_column is None:
            raise ValueError('Specified neuron IDs to '
                             'be ' + str(gdf_id_list) + ','
                             ' but file does not contain neuron IDs.')

        if t_start is None:
            raise ValueError('No t_start specified.')

        if t_stop is None:
            raise ValueError('No t_stop specified.')

        if not isinstance(t_start, pq.quantity.Quantity):
            raise TypeError('t_start (%s) is not a quantity.' % (t_start))

        if not isinstance(t_stop, pq.quantity.Quantity):
            raise TypeError('t_stop (%s) is not a quantity.' % (t_stop))

        # assert that no single column is assigned twice
        if id_column == time_column:
            raise ValueError('1 or more columns have been specified to '
                             'contain the same data.')

        # load GDF data
        f = open(self.filename)
        # read the first line to check the data type (int or float) of the spike
        # times, assuming that only the column of time stamps may contain
        # floats. then load the whole file accordingly.
        line = f.readline()
        if '.' not in line:
            data = np.loadtxt(self.filename, dtype=np.int32)
        else:
            data = np.loadtxt(self.filename, dtype=np.float)
        f.close()

        # check loaded data and given arguments
        if len(data.shape) < 2 and id_column is not None:
            raise ValueError('File does not contain neuron IDs but '
                             'id_column specified to ' + str(id_column) + '.')

        # get consistent dimensions of data
        if len(data.shape) < 2:
            data = data.reshape((-1, 1))

        # use only data from the time interval between t_start and t_stop
        data = data[np.where(np.logical_and(
                    data[:, time_column] >= t_start.rescale(
                        time_unit).magnitude,
                    data[:, time_column] < t_stop.rescale(time_unit).magnitude))]

        # create a list of SpikeTrains for all neuron IDs in gdf_id_list.
        # assign spike times to neuron IDs if id_column is given
        if id_column is not None:
            if gdf_id_list == []:
                gdf_id_list = np.unique(data[:, id_column]).astype(int)
                full_gdf_id_list = gdf_id_list
            else:
                full_gdf_id_list = \
                    np.unique(np.append(data[:, id_column].astype(int),
                                        np.asarray(gdf_id_list)))

            stdict = {i:[] for i in full_gdf_id_list}

            for i,nid in enumerate(data[:, id_column]):
                stdict[nid].append(data[i, time_column])

            spiketrain_list = []
            for nid in gdf_id_list:
                spiketrain_list.append(SpikeTrain(
                    np.array(stdict[nid]), units=time_unit,
                    t_start=t_start, t_stop=t_stop,
                    id=nid, **args))

        else:
            # if id_column is not given, all spike times are collected in one
            # spike train with id=None
            train = data[:, time_column]
            spiketrain_list = [SpikeTrain(train, units=time_unit,
                                          t_start=t_start, t_stop=t_stop,
                                          id=None, **args)]

        return spiketrain_list

    def read_segment(self, lazy=False, cascade=True,
                     gdf_id_list=None, time_unit=pq.ms, t_start=None,
                     t_stop=None, id_column=0, time_column=1, **args):
        """
        Read a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id_list : list or tuple, default: None
            Can be either list of GDF IDs of which to return SpikeTrain(s) or
            a tuple specifying the range (includes boundaries [start, stop])
            of GDF IDs. Must be specified if the GDF file contains neuron
            IDs, the default None then raises an error. Specify an empty
            list [] to retrieve the spike trains of all neurons with at least
            one spike.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), default: None
            Start time of SpikeTrain. t_start must be specified, the default None
            raises an error.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain for each ID in gdf_id_list.
        """

        if isinstance(gdf_id_list, tuple):
            if gdf_id_list[0] > gdf_id_list[1]:
                raise ValueError('second entry in range should be smaller or equal to first entry.')
            gdf_id_list = range(gdf_id_list[0], gdf_id_list[1] + 1)

        # __read_spiketrains() needs a list of IDs
        if gdf_id_list is None:
            gdf_id_list = [None]

        # create an empty Segment and fill in the spike trains
        seg = Segment()
        seg.spiketrains = self.__read_spiketrains(gdf_id_list,
                                                  time_unit, t_start,
                                                  t_stop,
                                                  id_column, time_column,
                                                  **args)

        return seg

    def read_spiketrain(
            self, lazy=False, cascade=True, gdf_id=None,
            time_unit=pq.ms, t_start=None, t_stop=None,
            id_column=0, time_column=1, **args):
        """
        Read SpikeTrain with specified neuron ID from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), default: None
            Start time of SpikeTrain. t_start must be specified.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """

        if (not isinstance(gdf_id, int)) and gdf_id is not None:
            raise ValueError('gdf_id has to be of type int or None')

        if gdf_id is None and id_column is not None:
            raise ValueError('No neuron ID specified but file contains '
                             'neuron IDs in column ' + str(id_column) + '.')

        # __read_spiketrains() needs a list of IDs
        return self.__read_spiketrains([gdf_id], time_unit,
                                       t_start, t_stop,
                                       id_column, time_column,
                                       **args)[0]
        

# TODO check documentation
