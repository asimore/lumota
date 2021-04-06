import time, logging
import os, sys, argparse
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import cpuinfo

import paho.mqtt.client as mqtt
import json
import socket

sys.path.append('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/src')
sys.path.append('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumota')


from libnyumaya import AudioRecognition,FeatureExtractor
from auto_platform import AudiostreamSource, play_command,default_libpath
from pixels import Pixels

REMOTE_SERVER = "www.google.com"
MQTT_BROKER = "ec2-52-37-146-89.us-west-2.compute.amazonaws.com"
PORT = 1883
KEEP_ALIVE = 10
flag_connected = 0
FIRMWARE_VERSION = "V_1.0"
CONFIRMATION_WAIT = 20
RECORDINGS_PATH = '/home/pi/projects/lumin/Lumin_FW_Src/audio_application/rec/'
import os
if not os.path.exists(RECORDINGS_PATH):
    os.makedirs(RECORDINGS_PATH)


#Reading CPU serial number on Linux based only
cpuserial = "0000000000000000"
info = cpuinfo.get_cpu_info()
arch = info['arch']
if arch.startswith('ARM'):
    f = open('/proc/cpuinfo','r')
    for line in f:
        if line[0:6]=='Serial':
           cpuserial = line[10:26]

DEV_UUID = "LUMIN_"+cpuserial
LOG = "ALL"
print("Device UUID : " + DEV_UUID)

#logging creation
log_file_name = "/var/log/lumin/Device_log.log"
logging.basicConfig(filename=log_file_name,format='%(asctime)s %(message)s',filemode='a')
logger=logging.getLogger()
logger.setLevel(logging.INFO)


phrases = {
    'help': ['help'],
    'intruder': ['intruder'],
    'fire': ['fire'],
    'yes': ['yes'],
    'no': ['no'],
    'hi': ['hi']
}

confirmation_message = "espeak --stdout 'Did you say {}' | aplay -Dsysdefault"

#Function to check internet connectivity, returns true is internet is up.
def check_internet(hostname):
	try:
		host = socket.gethostbyname(hostname)
		s = socket.create_connection((host,80),2)
		s.close()
		return True
	except:
		pass
	return False

#MQTT client connection callback function
def on_connect(client, userdata, flags, rc):
	global flag_connected
	flag_connected = 1
	print("MQTT Connect")
	logger.info("Connected to MQTT Broker")

#MQTT client disconnect callback function
def on_disconnect(client, userdata, rc):
	global flag_connected
	flag_connected = 0
	print("MQTT Disconnect")
	logger.error("Disconnected from MQTT Broker")

#Last will message JSON
last_will = {}
last_will['device_name'] = DEV_UUID
last_will['status'] = "offline"
last_will_json = json.dumps(last_will)

mqtt_client = mqtt.Client(DEV_UUID)
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.will_set("status", payload=last_will_json, qos=0, retain=True)

#MQTT message trigger function, sends an MQTT trigger with details in JSON format
def send_mqtt_trigger(time_stamp,trigger_name,confirmation):
    message = {}
    message['timestamp'] = time_stamp
    message['device_name'] = DEV_UUID
    message['sound_name'] = trigger_name
    message['is_confirmed'] = confirmation
    json_msg = json.dumps(message)
    print(json_msg)
    print("Sending MQTT trigger !")
    global flag_connected
    if(flag_connected == 0):
    	mqtt_client.connect(MQTT_BROKER,PORT,KEEP_ALIVE)
    	mqtt_client.loop_start()
    mqtt_client.publish("trigger",json_msg)


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def getModel(ARGS):
    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)

    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)

    #model.addHotWord('Fire', 10)
    #model.addHotWord('Intruder', 10)
    #model.addHotWord('Help', 10)
    #model.addHotWord('Yes', 10)
    #model.addHotWord('No', 10)

    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    return model


is_confirmed = False
is_recording = False
start_recording = True
stop_recording = False
newloop = True
pixels = Pixels()
pixels.on()

def confirmation():
    global is_confirmed

    is_confirmed = False
    pixels.on()
    print ('stopping confirmation wait {}: '.format(is_confirmed))

def main(ARGS):

    # os.system("espeak --stdout 'Starting the Service' | aplay -Dsysdefault")

    pixels.on()

    model = getModel(ARGS)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    wav_data_to_save = bytearray()
    global is_confirmed
    global is_recording
    global start_recording
    global stop_recording
    global newloop

    is_confirmed = False
    is_recording = False
    start_recording = True
    stop_recording = False
    newloop = True

    hotword = ""
    start = time.time()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            #pixels.on()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
            if is_recording:
                wav_data_to_save.extend(frame)
                pixels.recording()
                newloop = False
            newloop = True
        else:
            if spinner: spinner.stop()
            #pixels.on()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
            text = stream_context.finishStream()
            print ('Current recognition status {}: '.format(is_confirmed))
            newloop = True
            for p in phrases:
                for s in phrases[p]:
                    if s.upper() in text.upper():
                        if newloop and start_recording and not is_recording and p.upper() == 'HI':
                            print ("Recognized, p={} s={} newloop={} text={} Starting Recording".format(p, s, newloop, text.upper()))
                            os.system("espeak --stdout 'Starting Recording' | aplay -Dsysdefault")
                            is_recording = True
                            start_recording = False
                            stop_recording = True
                            newloop = False
                            pixels.recording()
                        elif stop_recording and is_recording and p.upper() == 'HI':
                            print ("Recognized, p={} s={} newloop={} text={} Stopping Recording".format(p, s, newloop, text.upper()))
                            # print ("Recognized, {} Stopping Recording".format(p))
                            print("Writing Audio")
                            vad_audio.write_wav(os.path.join(RECORDINGS_PATH, datetime.now().strftime("rec_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data_to_save)
                            os.system("espeak --stdout 'Recording Saved' | aplay -Dsysdefault")
                            wav_data_to_save = bytearray()
                            is_recording = False
                            start_recording = True
                            stop_recording = False
                            newloop = False
                            pixels.on()

                        if not is_recording:
                            if not is_confirmed and (p.upper() == 'FIRE' or p.upper() == 'INTRUDER' or p.upper() == 'HELP'):
                                t=threading.Timer(5.5,confirmation)
                                t.start()
                                pixels.detected()
                                os.system(confirmation_message.format(p))
                                is_confirmed = True
                                hotword = p
                                start = time.time()
                            elif is_confirmed and (p.upper() == 'YES'):
                                # send message
                                is_confirmed = False
                                t.join()
                                pixels.confirmed()
     #                           time.sleep(1)
                                if(check_internet(REMOTE_SERVER) == True):
                                    print ("Recognized, {}".format(p))
                                    now = datetime.now().isoformat()
                                    logger.info('Sending trigger...')
                                    send_mqtt_trigger(now,hotword,True)
                                    os.system("espeak --stdout 'Trigger sent' | aplay -Dsysdefault")
                                    is_confirmed = False
                                    pixels.on()
                                else:
                                    os.system("espeak --stdout 'No internet connection' | aplay")
                                    print("No internet connection, MQTT trigger not sent")
                                    logger.error("No internet connection, MQTT trigger not sent")
                                    pixels.ota()
                            elif is_confirmed and (p.upper() == 'NO'):
                               print ("Recognized, {}".format(p))
                               is_confirmed = False
                               t.join()
                               pixels.confirmed()
    #                           time.sleep(1)
                               pixels.on()
                            else:
                               print ("Recognized, {}".format(p))
                               #pixels.on()

            stream_context = model.createStream()

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
