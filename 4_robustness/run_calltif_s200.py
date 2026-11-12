"""Run CaLLTiF on the Schaefer-200 + Tian S1 (216-parcel) time series, two runs per graph.

Same settings as the main analysis (2_causal_graphs/); only the parcellation changes.
At 216 parcels and two runs (~304 samples) the lagged partial correlations are not
estimable, so this run provides the contemporaneous graphs only; see run_calltif_s200_4run.py.

Usage:
  python 4_robustness/run_calltif_s200.py --one        # time a single graph
  python 4_robustness/run_calltif_s200.py --workers 14 --conds nf,dc --studies 1,2,3,4,5
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'                       # 1 thread/worker -> no oversubscription
import sys
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
HERE = config.ROBUSTNESS_DIR
sys.path.insert(0, config.TIGRAMITE_DIR)
from tigramite.independence_tests import ParCorr   # noqa: E402
import tigramite.data_processing as pp              # noqa: E402
import calltif                                       # noqa: E402

NUM_PARCELS = 216
NODES = ['X%d' % i for i in range(1, NUM_PARCELS + 1)]
TAU_MIN, TAU_MAX, PC_ALPHA, NSAMP = 0, 1, 1, 152
ALPHA_BY_STUDY = {1: 1e-2 / 12, 2: 1e-2 / 4, 3: 1e-2 / 4, 4: 1e-2 / 4, 5: 1e-2 / 4}
DATA = os.environ.get('S200_DATA', os.path.join(HERE, 's200_data'))
OUT = os.environ.get('S200_OUT', os.path.join(HERE, 's200_results'))


def run_one(a0, a1, study):
    """Exactly the original inner loop."""
    data = {0: np.asarray(a0, dtype=np.float64)[-NSAMP:, :],
            1: np.asarray(a1, dtype=np.float64)[-NSAMP:, :]}
    df = pp.DataFrame(data, var_names=NODES, analysis_mode='multiple')
    return calltif.run_calltif(df, ParCorr(), tau_min=TAU_MIN, tau_max=TAU_MAX,
                               pc_alpha=PC_ALPHA, alpha_level=ALPHA_BY_STUDY[study])


def worker(task):
    study, cond, subj, ses, r, a0, a1, outpath = task
    if os.path.exists(outpath):
        return (outpath, 'skip', 0.0)
    try:
        t0 = time.time()
        link = run_one(a0, a1, study)
        np.savez(outpath, p_matrix=link['p_matrix'], val_matrix=link['val_matrix'],
                 graph=link.get('graph', np.array([])))
        return (outpath, 'done', time.time() - t0)
    except Exception as e:
        return (outpath, 'ERROR: %r' % e, 0.0)


def load_par(study, cond):
    f = os.path.join(DATA, f'study{study}_s200_{cond}.npz')
    return np.load(f, allow_pickle=True)['fmri_data_all_preprocessed_parcelated']


def build_tasks(studies, conds):
    tasks = []
    for study in studies:
        for cond in conds:
            par = load_par(study, cond)
            nsub, nses, nrun = par.shape[0], par.shape[1], par.shape[2]
            for subj in range(nsub):
                for ses in range(nses):
                    for r in range(nrun - 1):              # consecutive pairs r, r+1
                        a0 = np.asarray(par[subj][ses][r], dtype=np.float64)
                        a1 = np.asarray(par[subj][ses][r + 1], dtype=np.float64)
                        if a0.ndim != 2 or a1.ndim != 2:    # missing run -> skip pair
                            continue
                        out = os.path.join(OUT, f'{cond}_study{study}_subj{subj}_ses{ses}_pair{r}.npz')
                        tasks.append((study, cond, subj, ses, r, a0, a1, out))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one', action='store_true')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--conds', default='nf,dc')
    ap.add_argument('--studies', default='1,2,3,4,5')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.one:
        par = load_par(1, 'nf')
        a0, a1 = par[0][0][0], par[0][0][1]
        print('one-graph test: study1 nf subj0 ses0 pair0 | parcels', np.asarray(a0).shape[1], flush=True)
        t0 = time.time()
        link = run_one(a0, a1, 1)
        dt = time.time() - t0
        p, v = link['p_matrix'], link['val_matrix']
        a = ALPHA_BY_STUDY[1]
        print(f'DONE {dt:.1f}s ({dt/60:.2f} min) | p {p.shape} v {v.shape} | '
              f'val[{np.nanmin(v):.3f},{np.nanmax(v):.3f}] | sig(p<=alpha)={int((p<=a).sum())} '
              f'| lag0 sig={int((p[:,:,0]<=a).sum())} lag1 sig={int((p[:,:,1]<=a).sum())}', flush=True)
        np.savez(os.path.join(OUT, 'test_one_graph.npz'), p_matrix=p, val_matrix=v, seconds=dt)
        print('saved s200_results/test_one_graph.npz', flush=True)
        return

    studies = [int(x) for x in args.studies.split(',')]
    conds = args.conds.split(',')
    tasks = build_tasks(studies, conds)
    n = len(tasks)
    print(f'{n} graphs | {args.workers} workers | studies={studies} conds={conds}', flush=True)
    done = 0; t_start = time.time(); times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            outpath, status, dt = fut.result()
            done += 1
            if status == 'done':
                times.append(dt)
            if done % 20 == 0 or status.startswith('ERROR'):
                eta = (np.mean(times) * (n - done) / args.workers / 60) if times else 0
                print(f'[{done}/{n}] {status:8.8s} {os.path.basename(outpath)} '
                      f'({dt:.0f}s, elapsed {(time.time()-t_start)/60:.1f}m, ETA {eta:.0f}m)', flush=True)
    print(f'ALL DONE {(time.time()-t_start)/60:.1f} min | median/graph {np.median(times):.0f}s', flush=True)


if __name__ == '__main__':
    main()
