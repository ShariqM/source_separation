"""Convert WV1 files in LDC-WSJ0 to WAV files."""

import glob
import os
import multiprocessing
from functools import partial
import subprocess
import pdb

def load_src(name, fpath):
  import imp
  return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("constants", "../constants.py")
import constants


def get_spk_name(file_name):
  return file_name[:3]

def mp_worker(file_loc, dest_dir):
  file_name = file_loc.split('/')[-1]
  folder_name = get_spk_name(file_name)
  whole_dir = dest_dir + (folder_name + '/') * 2 # Use two folders (like in LibriSpeech)
  subprocess.call("mkdir -p %s" % whole_dir, shell=True)

  c = int(file_loc[-1])  # Channel
  wav_file_loc = whole_dir + file_name[:-4] + ".wv%d" % c
  tmp_2_file_loc = whole_dir + file_name[:-4] + "_tmp_2.wav"
  tmp_file_loc = whole_dir + file_name[:-4] + "_tmp.wav"
  subprocess.call("sph2pipe -f raw %s -f wav %s" % (file_loc, tmp_file_loc), shell=True)
  subprocess.call("ffmpeg -y -i %s -ar %d %s" % (tmp_file_loc, constants.Fs, tmp_2_file_loc), shell=True)

  subprocess.call("mv %s %s" % (tmp_2_file_loc, wav_file_loc), shell=True)
  subprocess.call("rm %s" % (tmp_file_loc), shell=True)
  print ("Process %s\tDONE" % file_loc)

def mp_handler():
  for c in (1,):  # Ignore channel 2 recording
    train_file_locs = glob.glob('data/wsj0_sph/*/wsj0/si_tr_s/*/*.wv%d' % c)
    train_dest_dir = 'data/wsj0/train/'
    pdb.set_trace()

    test_file_locs = glob.glob('data/wsj0_sph/*/wsj0/si_*t_05/*/*.wv%d' % c)
    test_dest_dir = 'data/wsj0/test/'

    both_file_locs =  (train_file_locs, test_file_locs)
    both_dest_dirs = (train_dest_dir, test_dest_dir)

    p = multiprocessing.Pool(2 * multiprocessing.cpu_count())
    for (file_locs, dest_dir) in zip(both_file_locs, both_dest_dirs):
      print ("Processing %d files." % len(file_locs))
      p.map(partial(mp_worker, dest_dir=dest_dir), file_locs)
      print ("*** Loop completed. ***")

if __name__ == '__main__':
  mp_handler()
