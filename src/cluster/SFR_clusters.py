import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate

min_mass = 1e3
max_mass = 2e4
alpha = 1.8
SF_max = 2e9
m82 = True
if m82:
    mw = False

if m82:
    name = "M82"
if mw:
    name = "MW"

# a few function definitions (now just used for plotting)
def cluster_pdf(x):
    return x ** -2

integral1 = integrate.quad(cluster_pdf, min_mass, max_mass)
A = 1 / integral1[0]

def cluster_pdf_norm(x):
    return A * x ** -2

def cluster_cdf(x):
    return x ** -1


#C = np.logspace(4, 6.7, 100000, endpoint=False)
#C = np.logspace(4, 6, 100000, endpoint=False)
#C = np.logspace(3, 5.7, 100000, endpoint=False)
#C = np.logspace(3, 4.7, 100000, endpoint=False)
C = np.logspace(np.log10(min_mass), np.log10(max_mass), 100000, endpoint=False)
#plt.loglog(C, cluster_pdf_norm(C))
#plt.show()

# below is an inverted cdf version of the sampling, only necessary if alpha = 1
#hist = cluster_pdf_norm(C)
#bin_edges = np.logspace(4,6.7,100001,endpoint=True)
#cum_values = np.zeros(bin_edges.shape)
#cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
#inv_cdf = interpolate.interp1d(cum_values, bin_edges)

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
while (total_SF < SF_max):
    cl = np.random.rand(1)
    #cl_mass = inv_cdf(cl)
    #cl_mass = sample_from_mass_CDF(cl, 2, 1e4, 5e6)
    #cl_mass = sample_from_mass_CDF(cl, 2, 1e4, 1e6)
    #cl_mass = sample_from_mass_CDF(cl, 2, 1e3, 5e4)
    cl_mass = sample_from_mass_CDF(cl, alpha, min_mass, max_mass)
    clusters = np.concatenate((clusters, cl_mass))
    total_SF += cl_mass
    tot_SF = np.concatenate((tot_SF, total_SF))
    phi = np.random.uniform(0, 2*np.pi, 1)
    phi_cl = np.concatenate((phi_cl, phi))
    z = np.random.uniform(-0.01, 0.01, 1)
    z_cl = np.concatenate((z_cl, z))

# plot the distribution of cluster masses (just to check
# that we're actually getting the pdf right)
print(np.size(clusters), np.size(np.where(clusters>1e4)))
#bins = np.logspace(4,6.7,20)
#bins = np.logspace(4,6,20)
#bins = np.logspace(3,6,30)
bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 20)
plt.hist(clusters, bins=bins, density=True)
#plt.hist(clusters, bins=bins, weights=clusters)
#plt.hist(clusters, bins=bins, density=True, cumulative=-1)
plt.plot(C, cluster_pdf_norm(C), color='k')
#plt.plot(C, 1e4*cluster_cdf(C), color='k')
plt.xscale('log')
plt.yscale('log')
#plt.ylim([1e4,1e9])
plt.xlabel('cluster mass [M$_\odot$]')
#plt.ylabel('mass in bin [M$_\odot$]')
plt.ylabel('dN / dM [M$_\odot^{-1}$]')
plt.savefig(f"cluster_masses_{name}.png", dpi=300)
plt.close()

# %%
print(np.sum(clusters))

# %%
#plt.plot(tot_SF)
#plt.show()

# %%
N_cl = np.size(clusters)

# now we'll specify the radial positions, this uses an exponential disk
# model with a scale radius given below
if m82:
    Rd = 0.8 # M82
if mw:
    Rd = 2.5 # MW
def f(R):
    return R * np.exp(- R / Rd)

if m82:
    integral = integrate.quad(f, 0, 4.5) # M82
if mw:
    integral = integrate.quad(f, 0, 9.0) # MW
A = 1 / integral[0]

# %%
def n(R):
    return A * R * np.exp(- R / Rd)

# generate radial distribution
# this uses the inverse cdf method to sample the distribution function
if m82:
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
plt.savefig(f"cluster_distribution_r_{name}.png", dpi=300)
plt.close()


#r = np.random.rand(N_cl)
#r_cl = inv_cdf(r)
#plt.hist(r_cl, bins=50, density=True)
#plt.plot(R, n(R), 'k')
#plt.show()

# %%
plt.polar(phi_cl, r_cl, 'k,', markersize=0.2)
plt.xticks([])
plt.savefig(f"cluster_distribution_phi_{name}.png", dpi=300)

# %%
x_cl = r_cl*np.cos(phi_cl)
y_cl = r_cl*np.sin(phi_cl)

# %%
output_arr = np.vstack((clusters, tot_SF, r_cl, phi_cl, z_cl)).T

# %%
np.shape(output_arr)

# %%
np.savetxt(f'cluster_list_{name}.txt', output_arr, fmt='%.5e', delimiter='\t', header='mass [M_sun]  total_SF [M_sun]  r [kpc]  phi [rad]  z [kpc]')
