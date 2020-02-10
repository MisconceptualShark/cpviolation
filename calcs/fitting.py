from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union,polygonize
from scipy.spatial import Delaunay
import shapely.geometry as geometry
import pylab as pl
import math

def exp_like(err,x,xt):
    like = 1
    for i in range(len(x)):
        like_i = (1/(np.sqrt(2*np.pi)*err[i]))*np.exp(-0.5*((x[i]-xt[i])/err[i])**2)
        like = like*like_i
    return like

def cov(x,y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x-xbar)*(y-ybar))/(len(x)-1)

def cov_mat(X):
    return np.array([[cov(X[0],X[0]),cov(X[0],X[1])], \
                     [cov(X[1],X[0]),cov(X[1],X[1])]])

def chisq_simp(obs,the,sige,sigt):
    '''
        chisq val, all parameters lists of values, simple
    '''
    N = len(obs)
    chi = 0
    for i in range(N):
        sig = np.sqrt(sige[i]**2 + sigt[i]**2)
        val = (obs[i]-the[i])/sig
        chi += val**2
    return chi

def chi_del(chi_min,chis,hs,ts):
    '''
        computes delta chisq, for several CLs? so 2 sigma so 95.45 just now
    '''
    delt_chis = (chis-chi_min)
    h_min,t_min = [],[]
    for i in range(len(hs)):
        if delt_chis[i] <= 5.99:
            h_min = np.append(h_min,hs[i])
            t_min = np.append(t_min,ts[i])

    return h_min, t_min

def chi_del2(chi_min,chis):
    '''
        computes delta chisq, for several CLs? so 2 sigma so 95.45 just now
    '''
    delt_chis = (chis-chi_min)
    hs = np.linspace(0,3.5,350)
    ts = np.linspace(-1,2,300)
    h_min,t_min = [],[]
    for i in range(len(hs)):
        for j in range(len(ts)):
            n = i*j
            if delt_chis[n] <= 5.99:
                h_min = np.append(h_min,hs[i])
                t_min = np.append(t_min,ts[j])

    return h_min, t_min

   
#def alpha_shape(points, alpha):
#    """
#    Compute the alpha shape (concave hull) of a set
#    of points.
#    @param points: Iterable container of points.
#    @param alpha: alpha value to influence the
#        gooeyness of the border. Smaller numbers
#        don't fall inward as much as larger numbers.
#        Too large, and you lose everything!
#    """
#    if len(points) < 4:
#        # When you have a triangle, there is no sense
#        # in computing an alpha shape.
#        return geometry.MultiPoint(list(points))
#           .convex_hull
#    def add_edge(edges, edge_points, coords, i, j):
#        """
#        Add a line between the i-th and j-th points,
#        if not in the list already
#        """
#            if (i, j) in edges or (j, i) in edges:
#                # already added
#                return
#            edges.add( (i, j) )
#            edge_points.append(coords[ [i, j] ])
#    coords = np.array([point.coords[0]
#                       for point in points])
#    tri = Delaunay(coords)
#    edges = set()
#    edge_points = []
#    # loop over triangles:
#    # ia, ib, ic = indices of corner points of the
#    # triangle
#    for ia, ib, ic in tri.vertices:
#        pa = coords[ia]
#        pb = coords[ib]
#        pc = coords[ic]
#        # Lengths of sides of triangle
#        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
#        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
#        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
#        # Semiperimeter of triangle
#        s = (a + b + c)/2.0
#        # Area of triangle by Heron's formula
#        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
#        circum_r = a*b*c/(4.0*area)
#        # Here's the radius filter.
#        #print circum_r
#        if circum_r < 1.0/alpha:
#            add_edge(edges, edge_points, coords, ia, ib)
#            add_edge(edges, edge_points, coords, ib, ic)
#            add_edge(edges, edge_points, coords, ic, ia)
#    m = geometry.MultiLineString(edge_points)
#    triangles = list(polygonize(m))
#    return cascaded_union(triangles), edge_points
#concave_hull, edge_points = alpha_shape(points,
#                                        alpha=1.87)
#_ = plot_polygon(concave_hull)
#_ = pl.plot(x,y,'o', color='#f16824')





