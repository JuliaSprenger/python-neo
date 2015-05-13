# -*- coding: utf-8 -*-
"""
Class for reading GDF data files.

For the user, it generates a :class:`Segment` or a :class:`Block` with a
sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
:class:`EventArray`.

For a developer, it is just an example showing guidelines for someone who wants
to develop a new IO module.

Depends on: scipy

Supported: Read

Author: sgarcia

"""

# needed for python 3 compatibility
from __future__ import absolute_import

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq

# but my specific IO can depend on many other packages
try:
    from scipy import stats
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err
else:
    HAVE_SCIPY = True
    SCIPY_ERR = None

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import Segment, AnalogSignal, SpikeTrain, EventArray


# I need to subclass BaseIO
class GdfIO(BaseIO):

    """
    Class for reading GDF files.

    For the user, it generates a :class:`Segment` or a :class:`Block` with a
    sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
    :class:`EventArray`.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the Neo IO API (:ref:`neo_io_API`)
      * Follow :ref:`io_guiline`

    Usage:
        >>> from neo import io
        >>> r = io.ExampleIO(filename='itisafake.nof')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print(seg.analogsignals)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ 0.19151945,  0.62399373,  0.44149764, ...,  0.96678374,
        ...
        >>> print(seg.spiketrains)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         [<SpikeTrain(array([ -0.83799524,   6.24017951,   7.76366686,   4.45573701,
            12.60644415,  10.68328994,   8.07765735,   4.89967804,
        ...
        >>> print(seg.eventarrays)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<EventArray: TriggerB@9.6976 s, TriggerA@10.2612 s, TriggerB@2.2777 s, TriggerA@6.8607 s, ...
        >>> anasig = r.read_analogsignal(lazy=True, cascade=False)
        >>> print(anasig._data_description)
        {'shape': (150000,)}
        >>> anasig = r.read_analogsignal(lazy=False, cascade=False)

    """

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [SpikeTrain]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects = [SpikeTrain]
    # This class is not able to write objects
    writeable_objects = []

    has_header = False
    is_streameable = False

    # do not supported write so no GUI stuff
    write_params = None

    name = 'gdf'

    extensions = ['gdf']

    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'fake'

    def __init__(self, filename=None):
        """


        Arguments:
            filename : the filename

        Note:
            - filename is here just for exampe because it will not be take in account
            - if mode=='dir' the argument should be dirname (See TdtIO)

        """
        BaseIO.__init__(self)
        self.filename = filename

# Segment reading is supported so I define this :
#     def read_segment(self,
# the 2 first keyword arguments are imposed by neo.io API
#                      lazy=False,
#                      cascade=True,
# all following arguments are decied by this IO and are
# free
#                      segment_duration=15.,
#                      num_analogsignal=4,
#                      num_spiketrain_by_channel=3,
#                      ):
#         """
#         Return a fake Segment.
#
#         The self.filename does not matter.
#
#         In this IO read by default a Segment.
#
#         This is just a example to be adapted to each ClassIO.
#         In this case these 3 paramters are  taken in account because this function
#         return a generated segment with fake AnalogSignal and fake SpikeTrain.
#
#         Parameters:
#             segment_duration :is the size in secend of the segment.
#             num_analogsignal : number of AnalogSignal in this segment
#             num_spiketrain : number of SpikeTrain in this segment
#
#         """
#
# sampling_rate = 10000.  # Hz
#         t_start = -1.
#
# time vector for generated signal
#         timevect = np.arange(
#             t_start, t_start + segment_duration, 1. / sampling_rate)
#
# create an empty segment
#         seg = Segment(name='it is a seg from exampleio')
#
#         if cascade:
# read nested analosignal
#             for i in range(num_analogsignal):
#                 ana = self.read_analogsignal(lazy=lazy, cascade=cascade,
#                                              channel_index=i, segment_duration=segment_duration, t_start=t_start)
#                 seg.analogsignals += [ana]
#
# read nested spiketrain
#             for i in range(num_analogsignal):
#                 for _ in range(num_spiketrain_by_channel):
#                     sptr = self.read_spiketrain(lazy=lazy, cascade=cascade,
#                                                 segment_duration=segment_duration, t_start=t_start, channel_index=i)
#                     seg.spiketrains += [sptr]
#
# create an EventArray that mimic triggers.
# note that ExampleIO  do not allow to acess directly to EventArray
# for that you need read_segment(cascade = True)
#             eva = EventArray()
#             if lazy:
# in lazy case no data are readed
# eva is empty
#                 pass
#             else:
# otherwise it really contain data
#                 n = 1000
#
# neo.io support quantities my vector use second for unit
#                 eva.times = timevect[
#                     (np.random.rand(n) * timevect.size).astype('i')] * pq.s
# all duration are the same
#                 eva.durations = np.ones(n) * 500 * pq.ms
# label
#                 l = []
#                 for i in range(n):
#                     if np.random.rand() > .6:
#                         l.append('TriggerA')
#                     else:
#                         l.append('TriggerB')
#                 eva.labels = np.array(l)
#
#             seg.eventarrays += [eva]
#
#         seg.create_many_to_one_relationship()
#         return seg

    def __read_spiketrains(self, data, gdf_id_list, time_unit,
                           t_start, t_stop, id_column=0,
                           time_column=1):
        '''Reads a list of spike trains with specified IDs from the GDF data.

        Parameters
        ----------
        data : numpy.array
            A Nx2 array containing the integer GDF data. The first column
            contains the neuron ID (or more general, event ID), the second
            column is the time stamp.
        gdf_id_list : list
            For each integer ID in this list the corresponding spike train
            (event train) is extracted from the data and returned. If None is
            specified, all spiketrains are returned. Default: None.
        time_unit : Quantity (time)
            The time unit of recorded time stamps.
        t_start : Quantity (time)
            Start time of the recorded GDF.
        t_stop : Quantity (time)
            Stop time of the recorded GDF.
        id_column : int
            Column index of neuron IDs
        time_column : int
            Column index of time stamps

        Returns
        -------
        spiketrain_list : list of SpikeTrain
            For each ID in gdf_id_list, one spike train is returned in this
            list. Duplicate entries in gdf_id_list result in duplicate
            SpikeTrain objects. Each SpikeTrain has an annotation id
            corresponding to its GDF ID.
        '''
        # list of spike trains
        spiketrain_list = []
        
        if len(data.shape) < 2 and id_column is not None:
            raise ValueError('File does not contain neuron IDs but '
                             'id_column specified to '+str(id_column)+'.')

        if time_column is None:
            raise ValueError('No spike times in file.')

        if None in gdf_id_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column '+str(id_column)+'.'
                             ' Specify empty list to ' 'retrieve'
                             ' spiketrains of all neurons.')

        if gdf_id_list != [None] and id_column is None:
            raise ValueError('Specified neuron IDs to '
                             'be '+str(gdf_id_list)+','
                             ' but file does not contain neuron IDs.')

        if t_stop is None:
            raise ValueError('No t_stop specified.')

        # assert that no single column is assigned twice
        if id_column == time_column and None not in [id_column,
                                                     time_column]:
            raise ValueError('1 or more columns have been specified to '
                             'contain the same data.')

        # get neuron gdf_id_list
        if gdf_id_list is []:
            gdf_id_list = np.unique(data[:, id_column]).astype(int)



        # assert that there are spike times in the file
        if time_column is None:
            raise ValueError('Time column is None. No spike times to \
be read in.')

        for i in gdf_id_list:
            # find the spike times for each neuron id
            if id_column is not None:
                train = data[np.where(data[:, id_column] == i), time_column][0]
            else:
                train = data[time_column]
            # create neo spike train
            spiketrain_list.append(SpikeTrain(
                train, units=time_unit, t_start=t_start, t_stop=t_stop,
                annotations={'id': i}))

        return spiketrain_list

    def read_segment(self, lazy=False, cascade=True,
                     gdf_id_list=None, time_unit=pq.ms, t_start=0.*pq.ms,
                     t_stop=None, id_column=0, time_column=1):
        '''Reads a segment containing of spike trains with specified IDs
        from the GDF data.

        Parameters
        ----------
        lazy : bool
        cascade : bool
        gdf_id_list : list
            For each integer ID in this list the corresponding spike train
            (event train) is extracted from the data and returned. If None is
            specified, all spiketrains are returned. Default: None.
        time_unit : Quantity (time)
            The time unit of recorded time stamps.
        t_start : Quantity (time)
            Start time of the recorded GDF.
        t_stop : Quantity (time)
            Stop time of the recorded GDF.
        id_column : int
            Column index of neuron IDs
        time_column : int
            Column index of time stamps

        Returns
        -------
        seg : Segment
            For each ID in gdf_id_list, one SpikeTrain is returned as part of
            the Segment. Duplicate entries in gdf_id_list result in duplicate
            SpikeTrain objects. Each SpikeTrain has an annotation id
            corresponding to its GDF ID.
        '''
        # load .gdf data
        try:
            data = np.loadtxt(self.filename, dtype=np.int32)
        except ValueError:
            data = np.loadtxt(self.filename, dtype=np.float)

        if gdf_id_list is None:
            gdf_id_list = [None]

        # create segment
        seg = Segment()
        seg.spiketrains = self.__read_spiketrains(data, gdf_id_list,
                                                  time_unit, t_start,
                                                  t_stop,
                                                  id_column=id_column,
                                                  time_column=time_column)
#        seg.create_relationships()

        return seg

    def read_spiketrain(
            self, lazy=False, cascade=True, gdf_id=None,
            time_unit=pq.ms, t_start=0 * pq.ms, t_stop=None,
            id_column=0, time_column=1):

        '''Reads SpikeTrain with specified ID from the GDF data.

        Parameters
        ----------
        lazy : bool
        cascade : bool
        gdf_id : int
            The GDF ID of the returned SpikeTrain.
        time_unit : Quantity (time)
            The time unit of recorded time stamps.
        t_start : Quantity (time)
            Start time of the recorded GDF.
        t_stop : Quantity (time)
            Stop time of the recorded GDF.
        id_column : int
            Column index of neuron IDs
        time_column : int
            Column index of time stamps

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id"
            corrsponding to the gdf_id parameter.
        '''
        # load .gdf data
        try:
            data = np.loadtxt(self.filename, dtype=np.int32)
        except ValueError:
            data = np.loadtxt(self.filename, dtype=np.float)

        # list of spike trains
        return self.__read_spiketrains(data, [gdf_id], time_unit,
                                       t_start, t_stop,
                                       id_column=id_column,
                                       time_column=time_column)[0]