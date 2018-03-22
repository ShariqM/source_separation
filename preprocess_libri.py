"""Convert FLAC files in LibriSpeech to WAV files."""
import glob
import multiprocessing
import subprocess
import pdb
import constants


def mp_worker(file_loc):
  wav_file_loc = file_loc[:-5] + ".wav"
  subprocess.call("ffmpeg -y -i %s -ar %d %s" % (file_loc, constants.Fs, wav_file_loc), shell=True)
  subprocess.call("rm -f %s" % (file_loc), shell=True)
  print ("Process %s\tDONE" % file_loc)

def mp_handler():
  file_locs = []
  file_locs.append(glob.glob('data/LibriSpeech/train-clean-100/*/*/*.flac'))
  file_locs.append(glob.glob('data/LibriSpeech/test-clean/*/*/*.flac'))

  for file_loc in file_locs:
    print ("Processing %d files." % len(file_loc))

    p = multiprocessing.Pool(2 * multiprocessing.cpu_count())
    p.map(mp_worker, file_loc)

if __name__ == '__main__':
  mp_handler()
