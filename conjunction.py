
from skyfield.api import load
import numpy as np
import pandas as pd
from tqdm import tqdm

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

planet_names_of_interest = ['sun', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']
conjunction_span_degrees = 45

ts = load.timescale(builtin=True)
t = ts.utc(1600, 1, range(0, 500*366, 1), 0, 0, 0)

df = pd.DataFrame(index=t.utc_datetime(), columns=planet_names_of_interest)

def in_conjunction(angles):
    sorted_angles = np.sort(angles.values)
    # sorted_angles_shifted = sorted_angles.take(range(1, len(angles)+1), mode='wrap')
    sorted_angles_shifted = np.roll(sorted_angles, -1)
    # print("sorted_angles = ", sorted_angles, "sorted_angles_shifted = ", sorted_angles_shifted)
    sorted_angles_diff = sorted_angles_shifted - sorted_angles
    sorted_angles_diff[-1] += 360.0
    sorted_angles_diff = list(map(lambda x: 360.0 - x if (x > 180.0) else x, sorted_angles_diff))
    # print(sorted_angles_diff)
    max_angle = max(sorted_angles_diff)
    return True if (max_angle < conjunction_span_degrees) else False

for pn in planet_names_of_interest: 
    print("Computing coordinates of {}".format(pn))
    observer = earth.at(t)
    planet = planets[pn]
    lat, lon, distance = observer.observe(planet).ecliptic_latlon()
    df.loc[:, pn] = lon.degrees

print("Calculating conjunctions...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    result = in_conjunction(row)
    df.at[index, 'in_conjunction'] = result

in_conjunction = df.loc[:, 'in_conjunction']
in_conjunction_shifted = np.roll(in_conjunction, 1)
df.loc[:, 'conjunction_change'] = in_conjunction ^ in_conjunction_shifted;
conjunction_change = df.loc[:, 'conjunction_change']

df.loc[:, 'conjunction_start'] = in_conjunction & conjunction_change;
df.loc[:, 'conjunction_end'] = np.invert(in_conjunction) & conjunction_change;

print("Post processing conjunction data ...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if (row['conjunction_start']):
        df.at[index, 'label'] = 'Conjunction Start'
    elif (row['conjunction_end']):
        df.at[index, 'label'] = 'Conjunction End'

print(df.loc[df['conjunction_change'] == True, planet_names_of_interest + ['label']])

# end of file
