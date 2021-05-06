"""
This class reads .set and .bin file data from the Axona acquisition system.

File format overview:
http://space-memory-navigation.org/DacqUSBFileFormats.pdf

In brief:
 data.set - setup file containing all hardware setups related to the trial
 data.bin - raw data file

There are many other data formats from Axona, which we do not consider (yet).
These are derivative from the raw continuous data (.bin) and could in principle
be extracted from it (see file format overview for details).

Author: Steffen Buergers

"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype,
                        _common_sig_characteristics)
import pathlib
import numpy as np
import os
import re
import contextlib
import datetime


class AxonaRawIO(BaseRawIO):
    """
    Class for reading raw, continuous data from the Axona dacqUSB system:
    http://space-memory-navigation.org/DacqUSBFileFormats.pdf

    The raw data is saved in .bin binary files with an accompanying
    .set file about the recording setup (see the above manual for details).

    Usage:
        import neo.rawio
        r = neo.rawio.AxonaRawIO(
            filename=os.path.join(dir_name, base_filename)
        )
        r.parse_header()
        print(r)
        raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                      i_start=0, i_stop=1024,  channel_names=channel_names)
        float_chunk = reader.rescale_signal_raw_to_float(
            raw_chunk, dtype='float64',
            channel_indexes=[0, 3, 6]
        )
    """

    extensions = ['bin', 'set']  # Never used?
    rawmode = 'multi-file'

    # In the .bin file, channels are arranged in a strange order.
    # This list takes a channel index as input and returns the actual
    # offset for the channel in the memory map (self._raw_signals).
    channel_memory_offset = [
        32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
        6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
        12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
        18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
        24, 25, 26, 27, 28, 29, 30, 31
    ]

    def __init__(self, filename):
        BaseRawIO.__init__(self)

        # Accepting filename with arbitrary suffix as input
        self.filename = pathlib.Path(filename).with_suffix('')
        self.set_file = self.filename.with_suffix('.set')
        self.bin_file = None
        self.tetrode_files = []

        # set file is required for all recordings
        if not self.set_file.exists():
            raise ValueError(f'Could not locate ".set" file. '
                             f'{self.filename.with_suffix(".set")} does not '
                             f'exist.')

        # detecting available files
        if self.filename.with_suffix('.bin').exists():
            self.bin_file = self.filename.with_suffix('.bin')

        for i in range(1, 33):
            unit_file = self.filename.with_suffix(f'.{i}')
            if unit_file.exists():
                self.tetrode_files.append(unit_file)
            else:
                break

        self.set_file_encoding = 'cp1252'

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        '''
        Read important information from .set header file, create memory map
        to raw data (.bin file) and prepare header dictionary in neo format.
        '''

        # Get useful parameters from .set file
        params = ['rawRate']
        global_params = self.get_header_parameters(self.set_file, params)

        unit_dtype = np.dtype([('spiketime', '>i4'), ('samples', 'int8', (50,))])
        # dt0 = [
            # ('file_id', 'S8'),
            # # label of sampling groun (e.g. "1kS/s" or "LFP Low")
            # ('label', 'S16'),
            # # number of 1/30000 seconds between data points
            # # (e.g., if sampling rate "1 kS/s", period equals "30")
            # ('period', 'uint32'),
            # ('channel_count', 'uint32')]

        # Useful num. bytes per continuous data packet
        self.file_parameters = {'bin': {'bytes_packet': 432,
                                        'bytes_data': 384,
                                        'bytes_head': 32,
                                        'bytes_tail': 16,
                                        'data_type': np.int16,
                                        'header_size': 0,
                                        'sampling_rate': int(global_params['rawRate']),
                                        'num_channels': len(self.get_active_tetrode()) * 4},
                                'set': {},
                                'unit': {'data_type': unit_dtype,
                                         'wf_left_sweep_us': 200,
                                         'tetrode_ids':[]}}



        signal_streams = []
        signal_channels = []
        if self.bin_file:
            # add derived parameters
            num_tot_packets = int(
                self.bin_file.stat().st_size / self.file_parameters['bin'][
                    'bytes_packet'])
            self.file_parameters['bin']['num_total_packets'] = num_tot_packets
            self.file_parameters['bin']['num_total_samples'] = num_tot_packets * 3

            # Create np.memmap to .bin file
            self._raw_signals = np.memmap(
                self.bin_file, dtype=self.file_parameters['bin']['data_type'],
                mode='r', offset=self.file_parameters['bin']['header_size']
            )

            signal_streams = self._get_signal_streams_header()
            signal_channels = self._get_signal_chan_header()

        self._raw_spikes = []
        spike_channels = []
        if self.tetrode_files:
            for i, tetrode_file in enumerate(self.tetrode_files):
                # collecting more unit parameters

                tetrode_dict = self.get_header_parameters(tetrode_file, None)
                tetrode_dict['timebase_hz'] = int(tetrode_dict['timebase'].replace(' hz',''))
                tetrode_dict['sample_rate_hz'] = int(tetrode_dict['sample_rate'].replace(' hz',''))
                tetrode_dict['num_chans'] = int(tetrode_dict['num_chans'])
                tetrode_dict.update(
                    {'header_size': self.get_header_lenth(tetrode_file),
                    'num_spikes': int(tetrode_dict['num_spikes']),
                    'wf_left_sweep': self.file_parameters['unit']['wf_left_sweep_us'] * tetrode_dict['timebase_hz'] * 10**-6})

                # memory mapping spiking data
                spikes = np.memmap(tetrode_file, dtype=self.file_parameters['unit']['data_type'],
                                   mode='r', offset=tetrode_dict['header_size'],
                                   shape=(tetrode_dict['num_spikes']))
                self._raw_spikes.append(spikes)

                for unit_id in range(1, tetrode_dict['num_chans']+1):
                    unit_name = f'tetrode-{i}_chan-{unit_id}'
                    unit_id = f'{i}-{unit_id}'
                    wf_units = 'dimensionless'
                    wf_gain = 1
                    wf_offset = 0.
                    wf_left_sweep = tetrode_dict['wf_left_sweep']
                    wf_sampling_rate = float(tetrode_dict['sample_rate_hz'])
                    spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                           wf_offset, wf_left_sweep,
                                           wf_sampling_rate))

                self.file_parameters['unit']['tetrode_ids'].append(i+1)
                self.file_parameters['unit'][i+1] = tetrode_dict

            # propagate common tetrode parameters to higher level
            tetrode_ids = self.file_parameters['unit']['tetrode_ids']
            if tetrode_ids:
                for key, value in self.file_parameters['unit'][tetrode_ids[0]].items():
                    if all([key in self.file_parameters['unit'][t] for t in tetrode_ids]) and \
                       all([self.file_parameters['unit'][t][key]==value for t in tetrode_ids]):
                        self.file_parameters['unit'][key] = value


        # Header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = np.array(signal_streams, dtype=_signal_stream_dtype)
        self.header['signal_channels'] = np.array(signal_channels, dtype=_signal_channel_dtype)
        self.header['spike_channels'] = np.array(spike_channels,
                                                 dtype=_spike_channel_dtype)
        self.header['event_channels'] = np.array([],
                                                 dtype=_event_channel_dtype)

        # Annotations
        self._generate_minimal_annotations()

        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann = bl_ann['segments'][0]
        seg_ann['rec_datetime'] = self.read_datetime()
        # sig_an = \
        #     seg_ann['signals'][0]['__array_annotations__']['tetrode_id'] = \
        #     [tetr for tetr in self.get_active_tetrode() for _ in range(4)]

    def _get_signal_streams_header(self):
        # create signals stream information (we always expect a single stream)
        return np.array([('stream 0', '0')], dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = 0.

        if 'num_total_packets' in self.file_parameters['bin']:
            t_stop = self.file_parameters['bin']['num_total_samples'] / \
                     self.file_parameters['bin']['sampling_rate']

        if 'unit' in self.file_parameters:
            # get tetrode recording durations in seconds
            tetrode_durations = [int(self.file_parameters['unit'][i]['duration'])
                                 for i in range(1, 33) if i in
                                 self.file_parameters['unit']]
            t_stop = max(t_stop, max(tetrode_durations))

        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self.file_parameters['bin']['num_total_samples']

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array (time x chan).
        Note that block_index and seg_index are always 1 (regardless of input).

        Raw data is in a single vector np.memmap with the following structure:

        Each byte packet (432 bytes) has header (32 bytes), footer (16 bytes)
        and three samples of 2 bytes each for 64 channels (384 bytes), which
        are jumbled up in a strange order. Each channel is remapped to a
        certain position (see get_channel_offset), and a channel's samples are
        allcoated as follows (example for channel 7):

        sample 1: 32b (head) + 2*38b (remappedID) and 2*38b + 1b (2nd byte)
        sample 2: 32b (head) + 128 (all chan. 1st entry) + 2*38b and ...
        sample 3: 32b (head) + 128*2 (all channels 1st and 2nd entry) + ...
        """

        # Set default values
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.file_parameters['bin']['num_total_samples']
        if channel_indexes is None:
            channel_indexes = [i for i in range(self.file_parameters['bin']['num_channels'])]

        num_samples = (i_stop - i_start)

        # Create base index vector for _raw_signals for time period of interest
        num_packets_oi = (num_samples + 2) // 3
        offset = i_start // 3 * (self.file_parameters['bin']['bytes_packet'] // 2)
        rem = (i_start % 3)

        sample1 = np.arange(num_packets_oi + 1, dtype=np.uint32) * \
            (self.file_parameters['bin']['bytes_packet'] // 2) + self.file_parameters['bin']['bytes_head'] // 2 + offset
        sample2 = sample1 + 64
        sample3 = sample2 + 64

        sig_ids = np.empty((sample1.size + sample2.size + sample3.size,),
                           dtype=sample1.dtype)
        sig_ids[0::3] = sample1
        sig_ids[1::3] = sample2
        sig_ids[2::3] = sample3
        sig_ids = sig_ids[rem:(rem + num_samples)]

        # Read one channel at a time
        raw_signals = np.ndarray(shape=(num_samples,
                                 len(channel_indexes)),
                                 dtype=self.file_parameters['bin']['data_type'])

        for i, ch_idx in enumerate(channel_indexes):

            chan_offset = self.channel_memory_offset[ch_idx]
            raw_signals[:, i] = self._raw_signals[sig_ids + chan_offset]

        return raw_signals


    def _spike_count(self, block_index, seg_index, unit_index):
        unit_id = self.header['spike_channels'][unit_index][1]
        tetrode_id, uid = np.asarray(unit_id.split('-'), dtype=int)

        raw_spikes = self._raw_spikes[tetrode_id]
        nb_tetrode_spikes = raw_spikes.shape[0]
        # adding one extra spike if last set of spikes is incomplete
        nb_unit_spikes = nb_tetrode_spikes / 4 + int((nb_tetrode_spikes % 4) >= uid)

        return nb_unit_spikes


    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        unit_id = self.header['spike_channels'][unit_index][1]
        tetrode_id, uid = np.asarray(unit_id.split('-'), dtype=int)

        raw_spikes = self._raw_spikes[tetrode_id]

        unit_spikes = raw_spikes['spiketime'][uid::4]

        # slice spike times only if needed
        if t_start is None or t_stop is None:

            if t_start is None:
                t_start = self._segment_t_start(block_index, seg_index)
            if t_stop is None:
                t_stop = self._segment_t_stop(block_index, seg_index)

            # convert to t_start and t_stop to sampling frequency
            # Note: this assumes no time offset!
            lim0 = t_start * self.file_parameters['unit'][uid]['timebase_hz']
            lim1 = t_stop * self.file_parameters['unit'][uid]['timebase_hz']

            # slice spike times
            mask = (unit_spikes >= lim0) & (unit_spikes <= lim1)
            spike_timestamps = unit_spikes[mask]

        else:
            spike_timestamps = unit_spikes

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self.file_parameters['unit']['timebase_hz']
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):

        assert block_index == 0
        assert seg_index == 0

        unit_id = self.header['spike_channels'][unit_index][1]
        tetrode_id, uid = np.asarray(unit_id.split('-'), dtype=int)

        raw_spikes = self._raw_spikes[tetrode_id]

        unit_spikes = raw_spikes['samples'][uid::4]

        waveforms = waveforms.reshape(nb_spike, 1, 50)





        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        # If there there is no waveform supported in the
        # IO them _spike_raw_waveforms must return None

        # In our IO waveforms come from all channels
        # they are int16
        # convertion to real units is done with self.header['spike_channels']
        # Here, we have a realistic case: all waveforms are only noise.
        # it is not always the case
        # we 20 spikes with a sweep of 50 (5ms)

        # trick to get how many spike in the slice
        ts = self._get_spike_timestamps(block_index, seg_index,
                                        spike_channel_index, t_start, t_stop)
        nb_spike = ts.size

        np.random.seed(2205)  # a magic number (my birthday)
        waveforms = np.random.randint(low=-2**4, high=2**4, size=nb_spike * 50, dtype='int16')
        waveforms = waveforms.reshape(nb_spike, 1, 50)
        return waveforms





    def get_header_lenth(self, file):
        """
        Scan file for the occurrence of 'data_start' and return the length
        of the header in bytes

        INPUT
        file (str or path): file to be loaded

        OUTPUT
        n_bytes (int): number of bytes occupied by the header
        """
        header = b''
        with open(file, 'rb') as f:
            for bin_line in f:
                if b'data_start' in bin_line:
                    header += b'data_start'
                    break
                else:
                    header += bin_line

        # adding arbitrary 12 bytes here -> need to confirm
        return len(header)

    # ------------------ HELPER METHODS --------------------
    # These are credited largely to Geoff Barrett from the Hussaini lab:
    # https://github.com/GeoffBarrett/BinConverter
    # Adapted or modified by Steffen Buergers

    def get_header_parameters(self, file, params):
        """
        Given a list of param., looks for each in first word of a phrase
        in the .set file. Adds found paramters as dictionary keys and
        following phrases as values (strings). If params is None all key
        are returned.

        INPUT
        file (str or path): file to be loaded
        params (list or set): parameter names to search for

        OUTPUT
        header (dict): dictionary with keys being the parameters that
                       were found & values being strings of the data.

        EXAMPLE
        self.get_header_parameters('file.set', ['experimenter', 'trial_time'])
        """
        header = {}
        if params is not None:
            params = set(params)
        with open(file, 'rb') as f:
            for bin_line in f:
                if b'data_start' in bin_line:
                    break
                line = bin_line.decode('cp1252').replace('\r\n', '').\
                    replace('\r', '').strip()
                parts = line.split(' ')
                key = parts[0]
                if (params is None) or (key in params):
                    header[key] = ' '.join(parts[1:])

        return header

    def get_active_tetrode(self):
        """
        Returns the ID numbers of the active tetrodes as a list.
        E.g.: [1,2,3,4] for a recording with 4 tetrodes (16 channels).
        """
        active_tetrodes = []

        with open(self.set_file, encoding=self.set_file_encoding) as f:
            for line in f:

                # The pattern to look for is collectMask_X Y,
                # where X is the tetrode number, and Y is 1 or 0
                if 'collectMask_' in line:
                    tetrode_str, tetrode_status = line.split(' ')
                    if int(tetrode_status) == 1:
                        tetrode_id = int(re.findall(r'\d+', tetrode_str)[0])
                        active_tetrodes.append(tetrode_id)

        return active_tetrodes

    def _get_channel_from_tetrode(self, tetrode):
        """
        This function will take the tetrode number and return the Axona
        channel numbers, i.e. Tetrode 1 = Ch1-Ch4, Tetrode 2 = Ch5-Ch8, etc.
        """
        return np.arange(1, 5) + 4 * (int(tetrode) - 1)

    def read_datetime(self):
        """
        Creates datetime object (y, m, d, h, m, s) from .set file header
        """
        with open(self.set_file, 'r', encoding=self.set_file_encoding) as f:
            for line in f:
                if line.startswith('trial_date'):
                    date_string = re.findall(r'\d+\s\w+\s\d{4}$', line)[0]
                if line.startswith('trial_time'):
                    time_string = line[len('trial_time') + 1::].replace('\n', '')

        return datetime.datetime.strptime(date_string + ', ' + time_string,
                                          "%d %b %Y, %H:%M:%S")

    def _get_channel_gain(self):
        """
        Read gain for each channel from .set file and return list of integers

        This is actually not the gain_ch value from the .set file, but the
        conversion factor from raw data to uV.

        Formula for .eeg and .X files, presumably also .bin files:

        1000*adc_fullscale_mv / (gain_ch*128)
        """
        gain_list = []

        with open(self.set_file, encoding='cp1252') as f:
            for line in f:
                if line.startswith('ADC_fullscale_mv'):
                    adc_fullscale_mv = int(line.split(" ")[1])
                if line.startswith('gain_ch'):
                    gain_list.append(
                        np.float32(re.findall(r'\d*', line.split(' ')[1])[0])
                    )

        return [1000 * adc_fullscale_mv / (gain * 128) for gain in gain_list]

    def _get_signal_chan_header(self):
        """
        Returns a 1 dimensional np.array of tuples with one entry per channel
        that recorded data. Each tuple contains the following information:

        channel name (1a, 1b, 1c, 1d, 2a, 2b, ...; num=tetrode, letter=elec),
        channel id (1, 2, 3, 4, 5, ... N),
        sampling rate,
        data type (int16),
        unit (uV),
        gain,
        offset,
        stream id
        """
        active_tetrode_set = self.get_active_tetrode()
        num_active_tetrode = len(active_tetrode_set)

        elec_per_tetrode = 4
        letters = ['a', 'b', 'c', 'd']
        dtype = self.file_parameters['bin']['data_type']
        units = 'uV'
        gain_list = self._get_channel_gain()
        offset = 0  # What is the offset?

        sig_channels = []
        for itetr in range(num_active_tetrode):

            for ielec in range(elec_per_tetrode):

                cntr = (itetr * elec_per_tetrode) + ielec
                ch_name = '{}{}'.format(itetr, letters[ielec])
                chan_id = str(cntr + 1)
                gain = gain_list[cntr]
                stream_id = '0'
                sig_channels.append((ch_name, chan_id, self.file_parameters['bin']['sampling_rate'], dtype,
                                     units, gain, offset, stream_id))

        return np.array(sig_channels, dtype=_signal_channel_dtype)
