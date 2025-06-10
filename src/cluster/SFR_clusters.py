import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from math import floor

m82 = False
high_z = False
mw = False
bursty_m82 = True
bursty_high_z = False
nuclear_burst = True  # same as bursty_m82 but with different burst periods

min_mass = 1e4
max_mass = 2.5e6
alpha = 1.9
if bursty_high_z or high_z:
    SFR_burst = 20  # M_sun / yr
if bursty_m82 or m82:
    SFR_burst = 5  # M_sun / yr
SFR_quiescent = 1
burst_period_duration = 40e6  # length of period when galaxy will form stars at a rate of SFR, yr
quiescent_period_duration = 20e6  # length of period when galaxy will form no new stars, yr
sim_duration = 2 * burst_period_duration + quiescent_period_duration  # total run time of simulation, yr, equivalent to two burst periods and one quiescent period
SF_max = SFR_quiescent * quiescent_period_duration + 2 * SFR_burst * burst_period_duration # M_sun, total mass of stars formed

if m82 or bursty_m82 or nuclear_burst:
    Rd = 0.3 # M82
if high_z or bursty_high_z:
    Rd = 0.8 # high_z
if mw:
    Rd = 2.5 # MW

if m82:
    name = "m82"
if mw:
    name = "MW"
if high_z:
    name = "high_z"
if bursty_high_z:
    name = "bursty_20"
if bursty_m82:
    name = "bursty_5"
if nuclear_burst:
    name = "nb"

# a few function definitions (now just used for plotting)
def cluster_pdf(x):
    return x ** -2

integral1 = integrate.quad(cluster_pdf, min_mass, max_mass)
A = 1 / integral1[0]

def cluster_pdf_norm(x):
    return A * x ** -2

def cluster_cdf(x):
    return x ** -1

C = np.logspace(np.log10(min_mass), np.log10(max_mass), 100000, endpoint=False)

# this is the function that is actually used to sample - it's an analytic solution for the
# pdf defined above, which is a powerlaw with the exponent -alpha
# here X is a random number between 0 and 1, that will be mapped to your function,
# alpha is the slope of the powerlaw, and mclmin and mclmax are the mininum and maximum
# cloud mass you want to sample
def sample_from_mass_CDF(X, alpha, mclmin, mclmax):
    return (mclmin**(-alpha+1) - (mclmin**(-alpha+1) - mclmax**(-alpha+1))*X )**(1/(-alpha+1))

# do the sampling from the mass function
# the "total_SF" number is how many total solar masses of clusters you want to form
# should be ~ your target SFR x how long you want to run the simulation
# also assign a random azimuthal position and z
clusters = np.empty(0)
tot_SF = np.empty(0)
phi_cl = np.empty(0)
z_cl = np.empty(0)
total_SF = 0

period_number = 0  # track whether it's a burst or quiescent period
period_end = burst_period_duration  # the time that the loop's current period of star formation/quiesence ends at
SFR = SFR_burst # set the SFR to the burst rate
period_end_mass = SFR_burst * burst_period_duration

time = [0]  # "simulation runtime", according to how much stellar mass has been formed

while (total_SF < SF_max):
    cl = np.random.rand(1)
    cl_mass = sample_from_mass_CDF(cl, alpha, min_mass, max_mass)
    clusters = np.concatenate((clusters, cl_mass))
    total_SF += cl_mass

    time.append(time[-1] + cl_mass/SFR)

    tot_SF = np.concatenate((tot_SF, total_SF))
    phi = np.random.uniform(0, 2*np.pi, 1)
    phi_cl = np.concatenate((phi_cl, phi))
    z = np.random.uniform(-0.01, 0.01, 1)
    z_cl = np.concatenate((z_cl, z))

    if tot_SF[-1] >= (period_end_mass):
        print(f"Period duration: {time[-1][0]/1e6:.1f} Myr, period_number: {period_number}, total_SF: {total_SF[0]:.2e} M_sun, number of clusters: {len(tot_SF)}")
        # if it's currently a burst formation period
        if (period_number == 0) or (period_number == 2):
            SFR = SFR_quiescent
            period_end_mass += quiescent_period_duration * SFR
        # if it's currently a quiescent period
        if (period_number == 1) or (period_number == 3):
            SFR = SFR_burst
            period_end_mass += burst_period_duration * SFR
        period_number += 1

# plot the distribution of cluster masses
print(f"Number of clusters: {np.size(clusters)}, number above 1e4 M_sun: {np.size(np.where(clusters>1e4))}")
bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 20)
plt.hist(clusters, bins=bins, density=True)
plt.plot(C, cluster_pdf_norm(C), color='k')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('cluster mass [M$_\odot$]')
plt.ylabel('dN / dM [M$_\odot^{-1}$]')
plt.savefig(f"{name}/cluster_masses_{name}.png", dpi=300)
plt.close()

# %%
print(f"Total cluster mass: {np.sum(clusters):.2e} M_sun")

# %%
plt.plot(np.array(time[1:])/1e6, tot_SF)
plt.xlabel("Time [Myr]")
plt.ylabel(r"Total SF [M$_\odot$]")
plt.show()
plt.savefig(f"{name}/total_sf_{name}.png", dpi=300)
plt.close()

# %%
N_cl = np.size(clusters)

# now we'll specify the radial positions, this uses an exponential disk
# model with a scale radius defined above
def f(R):
    return R * np.exp(- R / Rd)

if m82 or high_z or bursty_m82 or bursty_high_z or nb:
    integral = integrate.quad(f, 0, 4.5) # M82
if mw:
    integral = integrate.quad(f, 0, 9.0) # MW
A = 1 / integral[0]

# %%
def n(R):
    return A * R * np.exp(- R / Rd)

# generate radial distribution
# this uses the inverse cdf method to sample the distribution function
if m82 or high_z or bursty_m82 or bursty_high_z or nuclear_burst:
    R = np.linspace(0, 4.5, 1000, endpoint=False)+0.5*4.5/1000
    bin_edges = np.linspace(0,4.5,1001,endpoint=True)
if mw:
    R = np.linspace(0, 9.0, 1000, endpoint=False)+0.5*9.0/1000
    bin_edges = np.linspace(0,9.0,1001,endpoint=True)
hist = n(R)
cum_values = np.zeros(bin_edges.shape)
cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
inv_cdf = interpolate.interp1d(cum_values, bin_edges)

# plot radial distribution
r = np.random.rand(N_cl)
r_cl = inv_cdf(r)
plt.hist(r_cl, bins=50, density=True)
plt.plot(R, n(R), 'k')
plt.xlabel("radius [kpc]")
plt.savefig(f"{name}/cluster_distribution_r_{name}.png", dpi=300)
plt.close()

# %%
plt.polar(phi_cl, r_cl, 'k,', markersize=0.2)
plt.xticks([])
plt.savefig(f"{name}/cluster_distribution_phi_{name}.png", dpi=300)

# %%
x_cl = r_cl*np.cos(phi_cl)
y_cl = r_cl*np.sin(phi_cl)

# %%
output_arr = np.vstack((clusters, tot_SF, r_cl, phi_cl, z_cl)).T

# %%
np.shape(output_arr)

# %%
np.savetxt(f'{name}/cluster_list_{name}.txt', output_arr, fmt='%.5e', delimiter='\t', header='mass [M_sun]  total_SF [M_sun]  r [kpc]  phi [rad]  z [kpc]')
