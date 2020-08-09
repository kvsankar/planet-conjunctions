
from datetime import timedelta
import scipy.optimize
from skyfield.api import load
import numpy as np
import pandas as pd
from tqdm import tqdm

# constants

pi_degrees = 180.0
tau_degrees = 360.0

# configuration values of interest

planet_names_of_interest = ['sun', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']
conjunction_span_degrees = 45
search_interval_days = 1

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

eph = load('de422.bsp')

planets = {
        'sun': eph['sun'],
        'mercury': eph['mercury'],
        'venus': eph['venus'],
        'earth': eph['earth'],
        'moon': eph['moon'],
        'mars': eph['mars'],
        'jupiter': eph['jupiter barycenter'],
        'saturn': eph['saturn barycenter'],
        'uranus': eph['uranus barycenter'],
        'neptune': eph['neptune barycenter'],
        'pluto': eph['pluto barycenter'] }

earth = planets['earth']

def angle_span(angles):
    sorted_angles = np.sort(angles)
    sorted_angles_shift_left = np.roll(sorted_angles, -1)
    sorted_angles_diff = sorted_angles_shift_left - sorted_angles
    sorted_angles_diff[-1] += tau_degrees 
    sorted_angles_diff = list(map(lambda x: tau_degrees - x if (x > pi_degrees) else x, sorted_angles_diff))
    max_angle = max(sorted_angles_diff)
    return max_angle
    
def in_conjunction(angles):
    span = angle_span(angles.values)
    return (True, span) if (span < conjunction_span_degrees) else (False, span)

ts = load.timescale(builtin=True)
t = ts.utc(1600, 1, range(0, 500*366, search_interval_days), 0, 0, 0)

df = pd.DataFrame(index=t.utc_datetime(), columns=planet_names_of_interest + ['label'])

for pn in planet_names_of_interest: 
    print("Computing coordinates of {}".format(pn))
    observer = earth.at(t)
    planet = planets[pn]
    lat, lon, distance = observer.observe(planet).ecliptic_latlon()
    df.loc[:, pn] = lon.degrees

print("Calculating conjunctions...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    values = row[planet_names_of_interest]
    result, span = in_conjunction(values)
    df.at[index, 'span_degrees'] = span 
    df.at[index, 'in_conjunction'] = result

# In Conjunction Now              [False  False  True  True  True  False False]
# In Conjunction Earlier                 [False  False True  True  True  False  False]
# In Conjunction Later     [False  False  True   True  True  False False]
# Conjuction Start                [False  False  True  False False False]
# Conjuction End                  [False  False  False False True  False]

now_but_not_earlier = np.append([0], df['in_conjunction'].values[1:] & np.invert(df['in_conjunction'].values[:-1]))
now_but_not_later   = np.append(df['in_conjunction'].values[:-1] & np.invert(df['in_conjunction'].values[1:]), [0])

df.loc[:, 'conjunction_start'] = False 
df.loc[:, 'conjunction_end'] = False 

df.loc[now_but_not_earlier.astype(bool), 'conjunction_start'] = True 
df.loc[now_but_not_later.astype(bool),   'conjunction_end'] = True 

df.at[:, 'label'] = ''
df.loc[now_but_not_earlier.astype(bool), 'label'] += ['Start']
df.loc[now_but_not_later.astype(bool),   'label'] += ['End']

print("Calulting exact times...")

def f(jd):
    t = ts.tt(jd=jd)
    observer = earth.at(t)
    angles = []
    for pn in planet_names_of_interest: 
        planet = planets[pn]
        lat, lon, distance = observer.observe(planet).ecliptic_latlon()
        angles += [lon.degrees]
    return angle_span(angles) - conjunction_span_degrees
    
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if (row['conjunction_start']):
        t1_utc = index
        t0_utc = index - timedelta(days=search_interval_days)

        t0 = ts.from_datetime(t0_utc)
        t1 = ts.from_datetime(t1_utc)

        jd_conjunction = scipy.optimize.brentq(f, t0.tt, t1.tt)
        df.at[index, 'start_time'] = ts.tt(jd=jd_conjunction).utc_jpl()

    if (row['conjunction_end']):
        t0_utc = index
        t1_utc = index + timedelta(days=search_interval_days)

        t0 = ts.from_datetime(t0_utc)
        t1 = ts.from_datetime(t1_utc)

        jd_conjunction = scipy.optimize.brentq(f, t0.tt, t1.tt)
        df.at[index, 'end_time'] = ts.tt(jd=jd_conjunction).utc_jpl()

print(df.loc[df['label'] != '', planet_names_of_interest + ['label', 'start_time', 'end_time']])

        
# end of file
