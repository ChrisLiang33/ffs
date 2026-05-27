"""Per-scene ISSf comparison: reach, crash, timeout, time-to-reach in seconds."""
import csv

rows = list(csv.DictReader(open('overnight_summary.csv')))
scenes = ['in_dist','open','sparse','corridor','slalom','narrow','gauntlet']

print(f'{"scene":<10} {"phi":<8} {"reach%":<8} {"crash%":<8} {"timeo%":<8} {"t_reach_s":<10}')
print('-' * 60)
for s in scenes:
    issf = [(r['params'], float(r['reach'])*100, float(r['crash'])*100,
             float(r['timeout'])*100, r['mean_reach_steps'])
            for r in rows if r['method']=='issf' and r['scene']==s]
    issf.sort(key=lambda x: float(x[0].split('=')[1]) if 'phi=' in x[0] else 99)
    for p, r, c, t, t_steps in issf:
        t_sec = f'{float(t_steps)*0.02:5.2f}' if t_steps and t_steps != 'None' else '  n/a'
        print(f'{s:<10} {p:<8} {r:5.1f}    {c:5.1f}    {t:5.1f}    {t_sec}')
    print()
