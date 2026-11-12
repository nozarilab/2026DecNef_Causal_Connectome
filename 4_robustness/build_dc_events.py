"""Build the decoder-construction (DC) task-event templates used for the GLM task
regression (Supplementary Figs. 2-5 and 11).

The public DecNef Collection provides event files only for the neurofeedback (NF) runs.
The DC block design is fixed within each study, so one template per study is
reconstructed from the source publications and checked against the measured DC run
length (TR = 2 s): Study 1, 126 TR; Study 2, 152; Study 3, 152; Study 4, 154; Study 5, 307.
The GLM models the task-block structure, not stimulus identity, so one template per
study is applied to all subjects and runs.

Assumptions: the leading baseline of Study 3 is inferred from its run length. Study 5 is
a rapid design (600 images x 0.98 s per run) and is modelled as a regular image stream
with leading and trailing fixation.

Output: DecNef_events_DC/Study{N}/..._events.csv (onset, duration, event, trial; seconds).
"""
import os
import csv

OUT = 'DecNef_events_DC'
TR = 2.0


def write_template(study, rows, expected_tr):
    total = max(r[0] + r[1] for r in rows)
    d = os.path.join(OUT, f'Study{study}'); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'study{study}_task-decoder_events.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['onset', 'duration', 'event', 'trial'])
        for onset, dur, event, trial in rows:
            w.writerow([round(onset, 3), round(dur, 3), event, trial])
    ok = abs(total - expected_tr * TR) < 2.0
    print(f'Study{study}: {len(rows):4d} events, total={total:6.1f}s '
          f'(expected {expected_tr*TR:.0f}s={expected_tr}TR) {"OK" if ok else "MISMATCH"}  -> {path}')


def s1():  # Shibata: 10s fix + 20 x 12s (face .5 + rating 6 + report 5.5) + 2s fix = 252s
    r = [(0, 10, 'baseline', 'n/a')]
    for t in range(1, 21):
        s = 10 + (t - 1) * 12
        r += [(s, 0.5, 'face', t), (s + 0.5, 6, 'rating', t), (s + 6.5, 5.5, 'report', t)]
    r += [(250, 2, 'baseline', 'n/a')]
    write_template(1, r, 126)


def s2():  # Amano: 10s fix + 24 x 12s (grating 6 + black 6) + 6s fix = 304s
    r = [(0, 10, 'baseline', 'n/a')]
    for t in range(1, 25):
        s = 10 + (t - 1) * 12
        r += [(s, 6, 'grating', t), (s + 6, 6, 'blank', t)]
    r += [(298, 6, 'baseline', 'n/a')]
    write_template(2, r, 152)


def s3():  # Koizumi: 16s fix + 24 x 12s (fixation 6 + grating 6) = 304s  [16s leading split inferred]
    r = [(0, 16, 'baseline', 'n/a')]
    for t in range(1, 25):
        s = 16 + (t - 1) * 12
        r += [(s, 6, 'fixation', t), (s + 6, 6, 'grating', t)]
    write_template(3, r, 152)


def s4():  # Cortese: 20s fix + 16 x 18s (fix1+noise1+motion2+delay4+response4+ITI6) = 308s
    r = [(0, 20, 'baseline', 'n/a')]
    for t in range(1, 17):
        s = 20 + (t - 1) * 18
        r += [(s, 1, 'fixation', t), (s + 1, 1, 'noise', t), (s + 2, 2, 'stimulus', t),
              (s + 4, 4, 'delay', t), (s + 8, 4, 'response', t), (s + 12, 6, 'ITI', t)]
    write_template(4, r, 154)


def s5():  # Taschereau: ~13s fix + 600 images x 0.98s + ~13s fix ~= 614s (307 TR), rapid stream
    r = [(0, 13, 'baseline', 'n/a')]
    for k in range(600):
        r.append((round(13 + k * 0.98, 3), 0.98, 'image', k + 1))
    end = 13 + 600 * 0.98
    r += [(end, 614 - end, 'baseline', 'n/a')]
    write_template(5, r, 307)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for fn in (s1, s2, s3, s4, s5):
        fn()
    print('\nWrote one DC template per study under', OUT,
          '- apply to all subjects/runs (as the NF events are).')
