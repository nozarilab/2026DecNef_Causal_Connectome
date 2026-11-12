"""Run CaLLTiF on the GLM task-regressed residuals, for NF and DC sessions.

Same CaLLTiF settings as 2_causal_graphs/: tau_min = 0, tau_max = 1, pc_alpha = 1, ParCorr,
multiple-dataset mode with two consecutive runs per graph, last 152 samples per run,
alpha_level = 1e-2/12 for Study 1 and 1e-2/4 for Studies 2-5.
NF: 3 sessions x 8 runs -> 7 graphs per session. DC: 1 session x 6 runs -> 5 graphs.
One file per graph (p_matrix, val_matrix), so the run can be resumed:
  Results/glm_revision/glm_results_9subj/{cond}_{variant}_9subj_study{N}_subj{i}_ses{j}_pair{r}.npz

Usage:
  python 4_robustness/run_calltif_glm_9subj.py --workers 14 --conds dc,nf --variants hrf,binary --studies 1,2,3,4,5
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
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

NUM_PARCELS = 116
NODES = ['X%d' % i for i in range(1, NUM_PARCELS + 1)]
TAU_MIN, TAU_MAX, PC_ALPHA, NSAMP = 0, 1, 1, 152
ALPHA_BY_STUDY = {1: 1e-2 / 12, 2: 1e-2 / 4, 3: 1e-2 / 4, 4: 1e-2 / 4, 5: 1e-2 / 4}
DATA = os.environ.get('GLM9_DATA', os.path.join(HERE, 'glm_data_9subj'))
OUT = os.environ.get('GLM9_OUT', os.path.join(HERE, 'glm_results_9subj'))


def worker(task):
    study, cond, variant, subj, ses, r, a0, a1, outpath = task
    if os.path.exists(outpath):
        return (outpath, 'skip', 0.0)
    try:
        t0 = time.time()
        data = {0: a0.astype(np.float64), 1: a1.astype(np.float64)}
        df = pp.DataFrame(data, var_names=NODES, analysis_mode='multiple')
        link = calltif.run_calltif(df, ParCorr(), tau_min=TAU_MIN, tau_max=TAU_MAX,
                                   pc_alpha=PC_ALPHA, alpha_level=ALPHA_BY_STUDY[study])
        np.savez(outpath, p_matrix=link['p_matrix'], val_matrix=link['val_matrix'])
        return (outpath, 'done', time.time() - t0)
    except Exception as e:
        return (outpath, 'ERROR: %r' % e, 0.0)


def build_tasks(conds, variants, studies):
    tasks = []
    for cond in conds:
        for variant in variants:
            for study in studies:
                f = os.path.join(DATA, f'study{study}_{cond}_residuals_{variant}_9subj_flat.npz')
                if not os.path.exists(f):
                    print('missing', f, flush=True); continue
                d = np.load(f)
                by = {}
                for k in d.files:
                    if not k.startswith('run__'):
                        continue
                    _, subj, ses, r = k.split('__')
                    by.setdefault((int(subj), int(ses)), {})[int(r)] = d[k]
                for (subj, ses), runs in by.items():
                    for r in sorted(runs):
                        if r + 1 not in runs:
                            continue
                        a0, a1 = runs[r][-NSAMP:, :], runs[r + 1][-NSAMP:, :]   # unconditional [-152:]
                        out = os.path.join(OUT, f'{cond}_{variant}_9subj_study{study}_subj{subj}_ses{ses}_pair{r}.npz')
                        tasks.append((study, cond, variant, subj, ses, r, a0, a1, out))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--conds', default='dc,nf')          # dc first -> early DC previews
    ap.add_argument('--variants', default='hrf,binary')  # hrf first
    ap.add_argument('--studies', default='1,2,3,4,5')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    studies = [int(x) for x in args.studies.split(',')]
    tasks = build_tasks(args.conds.split(','), args.variants.split(','), studies)
    n = len(tasks)
    print(f'{n} graphs | {args.workers} workers | conds={args.conds} variants={args.variants} '
          f'studies={studies}', flush=True)
    done = 0; t0 = time.time(); times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            outpath, status, dt = fut.result()
            done += 1
            if status == 'done':
                times.append(dt)
            if done % 10 == 0 or status.startswith('ERROR'):
                eta = (np.mean(times) * (n - done) / args.workers / 60) if times else 0
                print(f'[{done}/{n}] {status:8.8s} {os.path.basename(outpath)} '
                      f'({dt:.0f}s, elapsed {(time.time()-t0)/60:.1f}m, ETA {eta:.0f}m)', flush=True)
    print(f'ALL DONE {(time.time()-t0)/60:.1f} min'
          + (f' | median/graph {np.median(times):.0f}s' if times else ''), flush=True)


if __name__ == '__main__':
    main()
