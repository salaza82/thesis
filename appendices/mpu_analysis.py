import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# Python base libraries
import os
import sys
import time
# Import general libraries with namespace
import matplotlib
# Necessary for computing on cluster
matplotlib.use("agg")
import numpy as np
import multiprocessing as mp
# Import HAWC software
sys.path.insert(1, os.path.join(os.environ['HOME'], 'hawc_software', 'threeml-analysis-scripts', 'fitModel'))
from analysis_modules import *
from threeML import *
from hawc_hal import HAL, HealpixConeROI
from threeML.minimizer.minimization import FitFailed
# Import Dark  Matter HAWC Libraries
import analysis_utils as au
import spectra as spec
import sources as srcs

#* READ ONLY PATHS This block will change eventually
MASS_LIST = './plotting/studies/nd/masses.txt'
CHAN_LIST = './plotting/studies/nd/chans.txt'

#* WRITE PATHS, default location is to scratch
OUT_PATH = os.path.join(os.environ["SCRATCH"], os.environ["USER"], 'New_duck')
print('Our out path is going to be {}'.format(OUT_PATH))

# Define parallel Function. Can also be run serially
def some_mass_fit(sigVs, datalist, shape, dSph, jfactor, mass_chan,
                  progress=None, log_file='', queue=None, i_job=0):

    if progress is None:
        progress = [0]
    else: # Create log files for each thread
        log_file = log_file.replace('.log', '_ThreadNo_')
        log_file = log_file + str(i_job) + ".log"
        sys.stdout = open(log_file, "w")

    fits = []

    try:
        for m_c in mass_chan:
            print(f'Mass chan tuple: {m_c}')
            mass = int(m_c[0])
            ch = m_c[1]
            # Build path to output files
            outPath = os.path.join(OUT_PATH, ch, dSph)
            au.ut.ensure_dir(outPath)

            if progress[i_job] < 0:
            # If the master gets a Keyboard interrupt, commit suicide.
                break

            ### Start Model Building for DM mass and SM channel ####
            spectrum = spec.DM_models.HDMSpectra()
            spectrum.set_channel(ch)

            myDwarf = ExtendedSource(dSph,spatial_shape=shape,
                                     spectral_shape=spectrum)

            spectrum.J = jfactor * u.GeV**2 / u.cm**5
            spectrum.sigmav = 1e-24 * u.cm**3 / u.s
            spectrum.set_dm_mass(mass * u.GeV)

            spectrum.sigmav.bounds = (1e-30,1e-12)
            model = Model(myDwarf)
            ### End model Building ####

            jl = JointLikelihood(model,datalist,verbose=False)

            try:
                result, lhdf = jl.fit(compute_covariance=False)
                ts = -2.*(jl.minus_log_like_profile(1e-30) - jl._current_minimum)
                # Also profile the LLH vs sv
                ll = jl.get_contours(spectrum.sigmav, sigVs[0],
                                     sigVs[-1], len(sigVs),
                                     progress=False, log=['False'])

                sigv_ul = au.ut.calc_95_from_profile(ll[0], ll[2])
                # Write results to file
                outFileLL = outPath+f"/LL_{dSph}_{ch}_mass{int(mass)}_GD.txt"
                np.savetxt(outFileLL, (sigVs, ll[2]),
                           delimiter='\t', header='sigV\tLL\n')

                with open(outPath+f"/results_{dSph}_{ch}_mass{int(mass)}_GD.txt", "w") as results_file:
                    results_file.write("mDM [GeV]\tsigV_95\tTS\tsigV_B.F.\n")

                    results_file.write("%i\t%.5E\t%.5E\t%.5E"%(mass, sigv_ul,
                                        ts, result.value[0]))
                # End write to file
            except FitFailed: # Don't kill all threads if a fit fails
                print("Fit failed. Go back and calculate this spectral model later")
                fits.append((ch, mass, -1, -1))
                with open(log_file+'.fail', 'w') as f_file:
                    f_file.write(f'{ch}, {mass}\n')

                progress[i_job] += 1
                matplotlib.pyplot.close() # Prevent leaky memory

            fits.append((ch, mass, result.value[0], ts))
            progress[i_job] += 1
            matplotlib.pyplot.close()
    except KeyboardInterrupt:
        progress[i_job] = -1

    fits = np.array(fits)
    if queue is None:
        return fits
    else:
        queue.put((i_job, fits))

def main(args):
    masses = np.loadtxt(MASS_LIST, dtype=int)
    chans = np.loadtxt(CHAN_LIST, delimiter=',', dtype=str)
    mass_chan = au.ut.permute_lists(chans, masses)

    print(f"DM masses for this study are: {masses}")
    print(f"SM Channels for this study are XX -> {chans}")
    print(mass_chan)

    # extract information from input argument
    dSph = args.dSph
    data_mngr = au.ut.Data_Selector('P5_NN_2D')
    dm_profile = srcs.Spatial_DM(args.catalog, dSph, args.sigma, args.decay)

    ### Extract Source Information ###
    if dm_profile.get_dec() < -22.0 or dm_profile.get_dec() > 62.0:
        raise ValueError("HAWC can't see this source D: Exitting now...")

    print(f'{dSph} information')
    print(f'jfac: {dm_profile.get_factor()}\tRA: {dm_profile.get_ra()}\tDec: {dm_profile.get_dec()}\n')

    shape = SpatialTemplate_2D(fits_file=dm_profile.get_src_fits())
    ### Finish Extract Source Information ###

    ### LOAD HAWC DATA ###
    roi = HealpixConeROI(data_radius=2.0, model_radius=8.0,
                         ra=dm_profile.get_ra(), dec=dm_profile.get_dec())
    bins = choose_bins.analysis_bins(args, dec=dm_profile.get_dec())

    hawc = HAL(dSph, data_mngr.get_datmap(), data_mngr.get_detres(), roi)
    hawc.set_active_measurements(bin_list=bins)
    datalist = DataList(hawc)
    ### FINISH LOAD HAWC DATA ###

    # set up SigV sampling. This sample is somewhat standardized
    sigVs = np.logspace(-28,-18, 1001, endpoint=True) # NOTE This will change whith HDM

    if args.n_threads == 1:
        # No need to start || programming just iterate over the masses
        kw_arg = dict(sigVs=sigVs, datalist=datalist, shape=shape, dSph=dSph,
                   jfactor=dm_profile.get_factor(), mass_chan=mass_chan,
                   log_file=args.log)
        some_mass_fit(**kw_arg)
    else:
        # I Really want to suppress TQMD output
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        x = np.array_split(mass_chan, args.n_threads)
        n_jobs = len(x)

        print("Thread jobs summary by mass and SM channel")
        for xi in x:
            print(f'{xi}')

        queue = mp.Queue()
        progress = mp.Array('i', np.zeros(n_jobs, dtype=int))

        # Define task pool that will be split amongsts threads
        kw_args = [dict(sigVs=sigVs, datalist=datalist, shape=shape,
                        dSph=dSph, jfactor=dm_profile.get_factor(),
                        mass_chan=mass_chan, progress=progress,
                        queue=queue, i_job=i, log_file=args.log)
                   for i, mass_chan in enumerate(x)]

        # Define each process
        procs = [mp.Process(target=some_mass_fit, kwargs=kw_args[i]) \
                 for i in range(n_jobs)]

        ### Start MASTER Thread only code block ###
        # Begin running all child threads
        for proc in procs: proc.start()

        try:
            # In this case, the master does nothing except monitor progress of the threads
            # In an ideal world, the master thread also does some computation.
            n_complete = np.sum(progress)
            while_count = 0

            while n_complete < len(mass_chan):

                if np.any(np.asarray(progress) < 0):
                    # This was no threads are stranded when killing the script
                    raise KeyboardInterrupt()
                if while_count%1000 == 0:
                    print(f"{np.sum(progress)} of {len(mass_chan)} finished")

                n_complete = np.sum(progress)
                time.sleep(.25)
                while_count += 1

        except KeyboardInterrupt:
        # signal to jobs that it's time to stop
            for i in range(n_jobs):
                progress[i] = -2
                print('\nKeyboardInterrupt: terminating early.')
        ### End MASTER Thread only code block ###

        fitss = [queue.get() for proc in procs]
        print(fitss)
        print(f'Thread statuses: {progress[:]}')

    # putting results in a file

    print("QUACK! All Done!")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description="Run a DM annihilation analysis on a dwarf spheroidal for a specific SM channel for DM masses [1 TeV to 10 PeV]")

    # Dwarf spatial modeling arguements
    p.add_argument("-ds", "--dSph", type=str,
                    help="dwarf spheroidal galaxy to be studied", required=True)
    p.add_argument("-cat", "--catalog", type=str, choices=['GS15', 'LS20'],
                   default='LS20', help="source catalog used")
    p.add_argument("-sig", "--sigma", type=int, choices=[-1, 0, 1], default=0,
                   help="Spatial model uncertainty. 0 corresponds to the median. +/-1 correspond to the +/1sigma uncertainty respectively.")

    # Arguements for the energy estimators
    p.add_argument("-e", "--estimator", type=str,
                   choices=['P5_NHIT', 'P5_NN_2D'],
                   default="P5_NN_2D", required=False,
                   help="The energy estimator choice. Options are: P5_NHIT, P5_NN_2D. GP not supported (yet).")
    p.add_argument("--use-bins", default=None, nargs="*",
                    help="Bins to  use for the analysis", dest="use_bins")
    p.add_argument('--select_bins_by_energy', default=None, nargs="*",
                   help="Does nothing. May fill in later once better understood")
    p.add_argument('--select_bins_by_fhit', default=None, nargs="*",
                   help="Also does nothing see above")
    p.add_argument( '-ex', "--exclude", default=None, nargs="*",
                   help="Exclude Bins", dest="exclude")

    # Computing and logging arguements.
    p.add_argument('-nt', '--n_threads', type=int, default=1,
                    help='Maximum number of threads spawned by script. Default is 4')
    p.add_argument('-log', '--log', type=str, required=True,
                    help='Name for log files. Especially needed for threads')

    p.add_argument('--decay', action="store_true",
                   help='Set spectral DM hypothesis to decay')

    args = p.parse_args()
    print(args.decay)
    if args.exclude is None: # default exclude bins 0 and 1
        args.exclude = ['B0C0Ea', 'B0C0Eb', 'B0C0Ec', 'B0C0Ed', 'B0C0Ee']

    if args.decay: OUT_PATH += '_dec'
    else: OUT_PATH += '_ann'

    OUT_PATH = OUT_PATH + '_' + args.catalog
    if args.sigma != 0: OUT_PATH += f'_{args.sigma}sig'

    main(args)
