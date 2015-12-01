#!/usr/bin/env python3

import argparse
import multiprocessing
import sys

# workaround possible backend bug
import matplotlib
matplotlib.use('agg')

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import freestream


class RepeatingFreeStreamer:
    def __init__(self, initial, grid_max, nframes, dt):
        self.initial = initial
        self.grid_max = grid_max
        self.nframes = nframes
        self.dt = dt

    def __call__(self, n):
        t = n*self.dt
        print('frame {:d} / {:d}, t = {:1.2f} fm'.format(n, self.nframes, t))
        fs = freestream.FreeStreamer(self.initial, self.grid_max, t)
        return t, fs.Tuv(0, 0)


def main():
    parser = argparse.ArgumentParser(
        description='animate a free streaming event',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_file', help='hdf5 input file')
    parser.add_argument('dataset', help='hdf5 dataset')
    parser.add_argument('output_file', help='mp4 output file')

    parser.add_argument('--grid-max', default=10.0, type=float,
                        help='grid xy max [fm]')
    parser.add_argument('--time-max', default=2.0, type=float,
                        help='animate until this time [fm/c]')
    parser.add_argument('--time-step', default=0.1, type=float,
                        help='timestep between frames [fm/c]')
    parser.add_argument('--fps', default=10, type=int,
                        help='frames per second')
    parser.add_argument('--cmap', default='Blues',
                        help='matplotlib colormap')
    parser.add_argument('--compress-cmap', default=1.0, type=float,
                        help='compress colormap range')

    args = parser.parse_args()

    try:
        cmap = getattr(plt.cm, args.cmap)
    except AttributeError:
        print('error: unknown cmap:', args.cmap, file=sys.stderr)
        exit(2)

    dt = args.time_step
    nframes = int(np.ceil(args.time_max/dt))

    with h5py.File(args.input_file, 'r') as f:
        initial = np.array(f[args.dataset])

    frames = [(0, initial)]
    rfs = RepeatingFreeStreamer(initial, args.grid_max, nframes, dt)

    with multiprocessing.Pool() as pool:
        frames += pool.map(rfs, range(1, nframes+1))

    plt.rcdefaults()

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes([0, 0, 1, 1], aspect='equal')
    ax.set_axis_off()

    xylim = -args.grid_max, args.grid_max

    ax.set_xlim(xylim)
    ax.set_ylim(xylim)

    img = ax.pcolorfast(xylim, xylim, np.array([[]]), cmap=cmap)
    txt = ax.text(0.95*xylim[0], 0.95*xylim[1], '', family='Lato', color='0.2',
                  ha='left', va='top')

    def init_func():
        return img, txt

    def draw_frame(frame):
        t, profile = frame
        img.norm.vmin = 0
        img.norm.vmax = profile.max() / args.compress_cmap
        img.set_data(profile)
        txt.set_text('{:1.2f} fm/c'.format(t))

        return img, txt

    print('writing', args.output_file)

    a = anim.FuncAnimation(fig, draw_frame, frames=frames,
                           init_func=init_func, blit=True)

    a.save(args.output_file, writer='ffmpeg', fps=args.fps,
           codec='h264', bitrate=2000)


if __name__ == "__main__":
    main()
