

# From https://stackoverflow.com/questions/43024371/determine-coordinates-at-conjunction-times/48256511#48256511 
# Thanks to Brandon Rhodes for Skyfield

import scipy.optimize
from skyfield.api import load, pi, tau

ts = load.timescale()
eph = load('de421.bsp')
sun = eph['sun']
earth = eph['earth']
venus = eph['venus']

# Every month from year 2000 to 2050.
t = ts.utc(2000, range(12 * 50))

# Where in the sky were Venus and the Sun on those dates?
e = earth.at(t)

lat, lon, distance = e.observe(sun).ecliptic_latlon()
sl = lon.radians

lat, lon, distance = e.observe(venus).ecliptic_latlon()
vl = lon.radians

# Where was Venus relative to the Sun?  Compute their difference in
# longitude, wrapping the value into the range [-pi, pi) to avoid
# the discontinuity when one or the other object reaches 360 degrees
# and flips back to 0 degrees.
relative_lon = (vl - sl + pi) % tau - pi

print(relative_lon)
print(relative_lon >= 0)
print((relative_lon >= 0)[:-1])
print(relative_lon < 0)
print((relative_lon < 0)[1:])

# Find where Venus passed from being ahead of the Sun to being behind.
conjunctions = (relative_lon >= 0)[:-1] & (relative_lon < 0)[1:]

print(conjunctions)

# For each month that included a conjunction, ask SciPy exactly when
# the conjunction occurred.

def f(jd):
    "Compute how far away in longitude Venus and the Sun are."
    t = ts.tt(jd=jd)
    e = earth.at(t)
    lat, lon, distance = e.observe(sun).ecliptic_latlon()
    sl = lon.radians
    lat, lon, distance = e.observe(venus).ecliptic_latlon()
    vl = lon.radians
    relative_lon = (vl - sl + pi) % tau - pi
    return relative_lon

for i in conjunctions.nonzero()[0]:
    t0 = t[i]
    t1 = t[i + 1]
    print("Starting search at", t0.utc_jpl())
    jd_conjunction = scipy.optimize.brentq(f, t[i].tt, t[i+1].tt)
    print("Found conjunction:", ts.tt(jd=jd_conjunction).utc_jpl())
    print()


