import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
from datetime import datetime
import noisereduce as nr
import soundfile as sf

fs = 16000       # INMP441 works at 48kHz
channels = 2              # stereo
M = 300
mu =0.125
blocksize = 512
external_channel = 1
internal_channel = 0


start_time = 0
w = np.zeros(M)
x_hist = np.zeros(M)
output_buffer = []
noise_buffer = []
noisy_buffer = []

def lms_callback(indata, outdata, frames, time, status):
	global w, x_hist, output_buffer

	x_block = indata[:, external_channel].astype(np.float32) / 32768.0
	d_block = indata[:, internal_channel].astype(np.float32) / 32768.0
	x_block = x_block / np.max(np.abs(x_block)) + 1e-9
	d_block = d_block / np.max(np.abs(d_block)) + 1e-9
	

	e_block = np.zeros_like(d_block)

	for i in range(frames):
		x_hist = np.roll(x_hist, -1)
		x_hist[-1] = x_block[i]

		y = np.dot(w, x_hist)
		e = d_block[i] - y
		norm_factor = np.dot(x_hist, x_hist) + 1e-6
		w += (2* mu* e* x_hist)/norm_factor
		e_block[i] = e

	output_buffer.append(e_block.copy())
	noise_buffer.append(x_block.copy())
	noisy_buffer.append(d_block.copy())

def save_audio(filename, buffer, fs):
	signal = np.concatenate(buffer)
	signal = np.clip(signal, -1.0, 1.0)
	int16_signal = (signal*32767).astype(np.int16)
	write(filename, fs, int16_signal)


try:
	start_time = time.time()
	print('Recording... Press CTRL+C to stop.')
	with sd.Stream(samplerate=fs, blocksize=blocksize, channels=2, dtype='int16', latency='high', callback=lms_callback):
		while True:
			sd.sleep(1000)

except KeyboardInterrupt:
	print()
	actual_duration = time.time() - start_time
	print('Stopped. Saving...')
	now = datetime.now()
	extension = now.strftime('%Y-%m-%d_%H-%M-%S') + '.wav'	
	output_buffer = output_buffer/ np.max(np.abs(output_buffer)) + 1e-9
	save_audio('lms_output' + extension, output_buffer, fs)
	save_audio('noisy' + extension, noisy_buffer, fs)
	save_audio('noise' + extension, noise_buffer, fs)
	duration = len(np.concatenate(noise_buffer))/fs
	print(f'Duration: {duration:.2f} seconds')
	print(f'Actual Duration: {actual_duration}')
	noise = np.concatenate(noise_buffer)
	noisy = np.concatenate(noisy_buffer)
	output = np.concatenate(output_buffer)

	diff = np.linalg.norm(noisy-output)
	
	print(f'filter diff {diff}')
	print(np.corrcoef(noise, noisy)[0,1])

	print('Saved')




