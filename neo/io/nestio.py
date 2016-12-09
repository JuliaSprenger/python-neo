# -*- coding: utf-8 -*-
"""
Class for reading output files of NEST.

Depends on: numpy, quantities

Supported: Read

Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os.path
import warnings
import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.io import GdfIO
from neo.core import Segment, SpikeTrain, AnalogSignalArray

value_type_dict = {'V': pq.mV,
                   'I': pq.pA,
                   'g': pq.CompoundUnit("10^-9*S"),
                   'no type': pq.dimensionless}


class NestIO(BaseIO):
    """
    Class for reading GDF files, e.g., the spike output of NEST.
    TODO 
    Usage:
        TODO
    """

    is_readable = True  # This class can only read data
    is_writable = False

    supported_objects = [SpikeTrain, AnalogSignalArray]
    readable_objects = [SpikeTrain, AnalogSignalArray]

    has_header = False
    is_streameable = False

    # do not supported write so no GUI stuff
    write_params = None

    name = 'nest'
    extensions = ['gdf', 'dat']
    mode = 'file'

    def __init__(self, filenames=None):
        """
        Parameters
        ----------
            filenames: string or list of strings, default=None
                The filename or list of filenames to load.
        """

        if isinstance(filenames, str):
            filenames = [filenames]

        self.filenames = filenames
        self.avail_formats = {}
        self.avail_IOs = {}

        for filename in filenames:
            path, ext = os.path.splitext(filename)
            ext = ext.strip('.')
            if ext in self.extensions:
                if ext in self.avail_IOs:
                    raise ValueError('Received multiple files with "%s" '
                                     'extention. Can only load single file of '
                                     'this type' % ext)
                self.avail_IOs[ext] = ColumnIO(filename)
            self.avail_formats[ext] = path

    def __read_analogsinalarrays(self, gid_list, time_unit, t_start=None,
                                 t_stop=None, sampling_period=None,
                                 id_column=0, time_column=1,
                                 value_columns=2, value_types=None,
                                 value_units=None, lazy=False):
        """
        Internal function called by read_analogsignalarray() and read_segment().
        """

        if 'dat' not in self.avail_formats:
            raise ValueError('Can not load analogsignalarrays. No dat file '
                             'provided.')

        # checking gid input parameters
        gid_list, id_column = self._check_input_gids(gid_list, id_column)
        # checking time input parameters
        t_start, t_stop = self._check_input_times(t_start, t_stop)

        # checking value input parameters
        (value_columns, value_types, value_units) = \
            self._check_input_values_parameters(value_columns, value_types,
                                                value_units)

        # defining standard column order for internal usage
        # [id_column, time_column, value_column1, value_column2, ...]
        column_ids = [id_column, time_column] + value_columns
        for i, cid in enumerate(column_ids):
            if cid is None:
                column_ids[i] = -1

        # assert that no single column is assigned twice
        column_list = [id_column, time_column] + value_columns
        if len(np.unique(column_list)) < 3:
            raise ValueError('1 or more columns have been specified to contain '
                             'the same data. Columns were specified to %s.'
                             '' % column_list)

        # extracting condition and sorting parameters for raw data loading
        (condition, condition_column,
         sorting_column) = self._get_conditions_and_sorting(id_column,
                                                            time_column,
                                                            gid_list,
                                                            t_start,
                                                            t_stop)
        # loading raw data columns
        data = self.avail_IOs['dat'].get_columns(
                column_ids=column_ids,
                condition=condition,
                condition_column=condition_column,
                sorting_columns=sorting_column)

        sampling_period = self._check_input_sampling_period(sampling_period,
                                                            time_column,
                                                            time_unit,
                                                            data)
        analogsignal_list = []

        if not lazy:
            # extracting complete gid list for anasig generation
            if (gid_list == []) and id_column is not None:
                gid_list = np.unique(data[:, id_column])

            # generate analogsignalarrays for each neuron ID
            for i in gid_list:
                selected_ids = self._get_selected_ids(i, id_column, time_column,
                                                      t_start, t_stop,
                                                      time_unit, data)

                # extract starting time of analogsignalarray
                if (time_column is not None) and data.size:
                    anasig_start_time = data[selected_ids[0], 1] * time_unit
                else:
                    # set t_start equal to sampling_period because NEST starts
                    # recording only after 1 sampling_period
                    anasig_start_time = 1. * sampling_period

                # create one analogsignalarray per value colum requested
                for v_id, value_column in enumerate(value_columns):
                    signal = data[selected_ids[0]:selected_ids[1], value_column]

                    # create AnalogSinalArray objects and annotate them with the
                    # neuron ID
                    analogsignal_list.append(AnalogSignalArray(
                            signal * value_units[v_id],
                            sampling_period=sampling_period,
                            t_start=anasig_start_time,
                            annotations={'id': i,
                                         'type': value_types[v_id]}))
                    # check for correct length of analogsignal
                    assert (analogsignal_list[-1].t_stop ==
                            anasig_start_time + len(signal) * sampling_period)
        return analogsignal_list

    def __read_spiketrains(self, gdf_id_list, time_unit,
                           t_start, t_stop, id_column,
                           time_column, **args):
        """
        Internal function called by read_spiketrain() and read_segment().
        """

        # assert that the file contains spike times
        if time_column is None:
            raise ValueError('Time column is None. No spike times to '
                             'be read in.')

        if None in gdf_id_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column ' + str(id_column) + '.'
                             ' Specify empty list to retrieve'
                             ' spike trains of all neurons.')

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
        filename = self.avail_formats['gdf'] + '.gdf'
        f = open(filename)
        # read the first line to check the data type (int or float) of the spike
        # times, assuming that only the column of time stamps may contain
        # floats. then load the whole file accordingly.
        line = f.readline()
        if '.' not in line:
            data = np.loadtxt(filename, dtype=np.int32)
        else:
            data = np.loadtxt(filename, dtype=np.float)
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

        # create a list of SpikeTrains for all neuron IDs in gdf_id_list
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

        # if id_column is not given, all spike times are collected in one
        # spike train with id=None
        else:
            train = data[:, time_column]
            spiketrain_list = [SpikeTrain(train, units=time_unit,
                                          t_start=t_start, t_stop=t_stop,
                                          id=None, **args)]

        return spiketrain_list


    def _check_input_times(self, t_start, t_stop):
        # checking input times
        if t_stop is None:
            t_stop = np.inf * pq.s
        if t_start is None:
            t_start = -np.inf * pq.s

        for time in (t_start, t_stop):
            if not isinstance(time, pq.quantity.Quantity):
                raise TypeError('Time value (%s) is not a quantity.' % time)
        return t_start, t_stop

    def _check_input_values_parameters(self, value_columns, value_types,
                                       value_units):
        if value_columns is None:
            raise ValueError('No value column provided.')
        if isinstance(value_columns, int):
            value_columns = [value_columns]
        if value_types is None:
            value_types = ['no type'] * len(value_columns)
        elif isinstance(value_types, str):
            value_types = [value_types]

        # translating value types into units as far as possible
        if value_units is None:
            short_value_types = [vtype.split('_')[0] for vtype in value_types]
            if not all([svt in value_type_dict for svt in short_value_types]):
                raise ValueError('Can not interpret value types '
                                 '"%s"' % value_types)
            value_units = [value_type_dict[svt] for svt in short_value_types]

        # checking for same number of value types, units and columns
        if not (len(value_types) == len(value_units) == len(value_columns)):
            raise ValueError('Length of value types, units and columns does '
                             'not match (%i,%i,%i)' % (len(value_types),
                                                       len(value_units),
                                                       len(value_columns)))
        if not all([isinstance(vunit, pq.UnitQuantity) for vunit in
                    value_units]):
            raise ValueError('No value unit or standard value type specified.')

        return value_columns, value_types, value_units

    def _check_input_gids(self, gid_list, id_column):
        if gid_list is None:
            gid_list = [gid_list]

        if None in gid_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column %s. Specify empty list to '
                             'retrieve spiketrains of all neurons.'
                             '' % str(id_column))

        if gid_list != [None] and id_column is None:
            raise ValueError('Specified neuron IDs to be %s, but no ID column '
                             'specified.' % gid_list)
        return gid_list, id_column

    def _check_input_sampling_period(self, sampling_period, time_column,
                                     time_unit, data):
        if sampling_period is None:
            if time_column is not None:
                data_sampling = np.unique(
                    np.diff(sorted(np.unique(data[:, 1]))))
                if len(data_sampling) > 1:
                    raise ValueError('Different sampling distances found in '
                                     'data set (%s)' % data_sampling)
                else:
                    dt = data_sampling[0]
            else:
                raise ValueError('Can not estimate sampling rate without time '
                                 'column id provided.')
            sampling_period = pq.CompoundUnit(str(dt) + '*'
                                              + time_unit.units.u_symbol)
        elif not isinstance(sampling_period, pq.UnitQuantity):
            raise ValueError("sampling_period is not specified as a unit.")
        return sampling_period

    def _get_conditions_and_sorting(self, id_column, time_column, gid_list,
                                    t_start, t_stop):
        condition, condition_column = None, None
        sorting_column = []
        if ((gid_list is not [None]) and (gid_list is not None)):
            if gid_list != []:
                condition = lambda x: x in gid_list
                condition_column = id_column
            sorting_column.append(0)  # Sorting according to gids first
        if time_column is not None:
            sorting_column.append(1)  # Sorting according to time
        elif t_start != -np.inf and t_stop != np.inf:
            warnings.warn('Ignoring t_start and t_stop parameters, because no '
                          'time column id is provided.')
        if sorting_column == []:
            sorting_column = None
        else:
            sorting_column = sorting_column[::-1]
        return condition, condition_column, sorting_column

    def _get_selected_ids(self, gid, id_column, time_column, t_start, t_stop,
                          time_unit, data):
        gid_ids = np.array([0, data.shape[0]])
        if id_column is not None:
            gid_ids = np.array([np.searchsorted(data[:, 0], gid, side='left'),
                                np.searchsorted(data[:, 0], gid, side='right')])
        gid_data = data[gid_ids[0]:gid_ids[1], :]

        # select only requested time range
        id_shifts = np.array([0, 0])
        if time_column is not None:
            id_shifts[0] = np.searchsorted(gid_data[:, 1],
                                           t_start.rescale(time_unit).magnitude,
                                           side='left')
            id_shifts[1] = (np.searchsorted(gid_data[:, 1],
                                            t_stop.rescale(time_unit).magnitude,
                                            side='left') - gid_data.shape[0])

        selected_ids = gid_ids + id_shifts
        return selected_ids

    def read_segment(self, gid_list=None, time_unit=pq.ms, t_start=None,
                     t_stop=None, sampling_period=None, id_column=0,
                     time_column=1, value_columns=2, value_types=None,
                     value_units=None, lazy=False, cascade=True):
        """
        Read a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Parameters
        ----------
        gid_list : list, default: None
            A list of GDF IDs of which to return SpikeTrain(s). gid_list must
            be specified if the GDF file contains neuron IDs, the default None
            then raises an error. Specify an empty list [] to retrieve the spike
            trains of all neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        sampling_period : Quantity (frequency), optional, default: None
            Sampling period of the recorded data.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.
        value_columns : int, optional, default: 2
            Column index of the analog values recorded.
        value_types : str, optional, default: None
            Nest data type of the analog values recorded, eg.'V_m', 'I', 'g_e'
        value_units : Quantity (amplitude), default: None
            The physical unit of the recorded signal values
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain and one AnalogSignalArray for
            each ID in gid_list.
        """
        if isinstance(gid_list, tuple):
            if gid_list[0] > gid_list[1]:
                raise ValueError('second entry in range should be '
                                 'greater or equal to first entry.')
            gid_list = range(gid_list[0], gid_list[1] + 1)

        # __read_xxx() needs a list of IDs
        if gid_list is None:
            gid_list = [None]

        # create an empty Segment
        seg = Segment()

        if cascade:
            ###################################
            # WARNING: This only works if column structure of gdf and dat
            # file is
            # identical
            ###################################
            # seg = super(NestIO, self).read_segment(lazy=lazy, cascade=cascade,
            #                                        gdf_id_list=gid_list,
            #                                        time_unit=time_unit,
            #                                        t_start=t_start,
            #                                        t_stop=t_stop,
            #                                        id_column=id_column,
            #                                        time_column=time_column)

            # Load analogsignalarrays and attach to Segment
            if 'dat' in self.avail_formats:
                seg.analogsignalarrays = self.__read_analogsinalarrays(
                        gid_list,
                        time_unit,
                        t_start,
                        t_stop,
                        sampling_period=sampling_period,
                        id_column=id_column,
                        time_column=time_column,
                        value_columns=value_columns,
                        value_types=value_types,
                        value_units=value_units,
                        lazy=lazy)
            if 'gdf' in self.avail_formats:
                seg.spiketrains = self.__read_spiketrains(gid_list,
                                                          time_unit,
                                                          t_start,
                                                          t_stop,
                                                          id_column=id_column,
                                                          time_column=time_column)


        return seg

    def read_analogsignalarray(self, lazy=False,
                               gid=None, time_unit=pq.ms, t_start=None,
                               t_stop=None, sampling_period=None, id_column=0,
                               time_column=1, value_column=2, value_type=None,
                               value_unit=None):
        """
        Read AnalogSignalArray with specified neuron ID from the DAT data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        gid : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs, the default None then raises an
            error. Specify an empty list [] to retrieve the spike trains of all
            neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        sampling_period : Quantity (frequency), optional, default: None
            Sampling period of the recorded data.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.
        value_column : int, optional, default: 2
            Column index of the analog values recorded.
        value_type : str, optional, default: None
            Nest data type of the analog values recorded, eg.'V_m', 'I', 'g_e'
        value_unit : Quantity (amplitude), default: None
            The physical unit of the recorded signal values

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """

        # __read_spiketrains() needs a list of IDs
        return self.__read_analogsinalarrays([gid], time_unit,
                                             t_start, t_stop,
                                             sampling_period=sampling_period,
                                             id_column=id_column,
                                             time_column=time_column,
                                             value_columns=value_column,
                                             value_types=value_type,
                                             value_units=value_unit,
                                             lazy=lazy)[0]

    def read_spiketrain(
            self, lazy=False, cascade=True, gdf_id=None,
            time_unit=pq.ms, t_start=None, t_stop=None,
            id_column=0, time_column=1, **args):
        """
        Read a SpikeTrain with specified neuron ID from the GDF data.

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
            raise ValueError('gdf_id has to be of type int or None.')

        if gdf_id is None and id_column is not None:
            raise ValueError('No neuron ID specified but file contains '
                             'neuron IDs in column ' + str(id_column) + '.')

        # __read_spiketrains() needs a list of IDs
        return self.__read_spiketrains([gdf_id], time_unit,
                                       t_start, t_stop,
                                       id_column, time_column,
                                       **args)[0]



class ColumnIO:
    def __init__(self, filename):

        self.filename = filename

        # read the first line to check the data type (int or float) of the data
        f = open(self.filename)
        line = f.readline()

        additional_parameters = {}
        if '.' not in line:
            additional_parameters['dtype'] = np.int32

        self.data = np.loadtxt(self.filename, **additional_parameters)

    def get_columns(self, column_ids='all', condition=None,
                    condition_column=None, sorting_columns=None):
        """
        :param column_ids:
        :param condition:
        :param condition_column:
        :param sorting_columns: Column ids to sort by. In increasing sorting
        priority!
        :return:
        """

        if column_ids == [] or column_ids == 'all':
            column_ids = range(self.data.shape[-1])

        if isinstance(column_ids, (int, float)):
            column_ids = [column_ids]
        column_ids = np.array(column_ids)

        if column_ids is not None:
            if max(column_ids) >= len(self.data) - 1:
                raise ValueError('Can not load column ID %i. File contains '
                                 'only %i columns' % (max(column_ids),
                                                      len(self.data)))

        if sorting_columns is not None:
            if isinstance(sorting_columns, int):
                sorting_columns = [sorting_columns]
            if max(sorting_columns) >= len(self.data) - 1:
                raise ValueError('Can not sort by column ID %i. File contains '
                                 'only %i columns' % (max(column_ids),
                                                      len(self.data)))

        # Starting with whole dataset being selected for return
        selected_data = self.data

        # Apply filter condition to rows
        if condition and (condition_column is None):
            raise ValueError('Filter condition provided, but no '
                             'condition_column ID provided')
        elif (condition_column is not None) and (condition is None):
            warnings.warn('Condition column ID provided, but no condition '
                          'given. No filtering will be performed.')

        elif (condition is not None) and (condition_column is not None):
            condition_function = np.vectorize(condition)
            mask = condition_function(selected_data[:,
                                      condition_column]).astype(bool)

            selected_data = selected_data[mask, :]

        # Apply sorting if requested
        if sorting_columns is not None:
            values_to_sort = selected_data[:, sorting_columns].T
            ordered_ids = np.lexsort(tuple(values_to_sort[i] for i in
                                           range(len(values_to_sort))))
            selected_data = selected_data[ordered_ids, :]

        # Select only requested columns
        selected_data = np.squeeze(selected_data[:, column_ids])

        return selected_data

        # Alternative implementation idea: Index all data rows during
        # initialization and reimplement get_columns based on this

        # def create_index_for_column(self,column_id):
        #     if column_id > self.data.shape[1]:
        #         raise ValueError('Column index out of range of data sets')
        #
        #     if column_id not in self.indexed_columns:
        #         contained_values = np.unique(self.data[:,column_id])
        #
        #     else:
        #         return self.indexed_columns[column_id]
