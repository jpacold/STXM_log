# Usage: python STXM_log.py [directory]
# If no directory is specified, the default
# is the data directory corresponding to
# the current date.
# 
# Scrapes the directory and uses the scan parameters
# taken from .hdr files to generate a logbook (.pdf).
#
#
# This is meant mainly to give an overview of the
# entire day of data, for example to let staff
# diagnose a problem. It has some differences from
# STXM_log.py, which is meant to produce a shorter
# and more easily readable log of data collection
# from one sample:
#
#  STXM_log                                   STXM_log_live
#  ------------------------------------       ------------------------------------------
#  All saved scans included                   Only image scans, line scans, and point scans included
#  Filename is fixed by date                  Filename set by user
#  Uses coordinates in .hdr files             Tries to correct coordinates in .hdr files for
#         to generate displays                       sample drift using image cross-correlation
#  Elemental maps not shown                   Interprets 2-image NEXAFS scans as elemental maps
#  12 panels per page                         6 panels per page (easier to paste a hard copy into a notebook)
#
#

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib import cm

import datetime
from time import localtime, strftime
from sys import argv
from os import path
from glob import glob


class scan():
    """Class that holds data on a single scan.
    As applicable, this may include:
        scantype, energies, dimensions, position,
        one image, map, associated spectrum files
        
    Attributes:
        scantype
            May be one of the following:
            Image Scan
            NEXAFS Image Scan
            Focus Scan
            OSA Scan
            OSA Focus Scan
            NEXAFS Line Scan
            Detector Scan
            NEXAFS Point Scan
        
        energies
            List of energies (one element long
            if the scan is not a NEXAFS scan)
        
        xcenter, ycenter, xmin, ymin, xmax, ymax
            Position data
        
        xstep, ystep
            Pixel size (for image scans)
        
        img
            Array of pixel values (for all
            types except NEXAFS Point Scan)
    
    Methods:
        contains(pt)
            For image scans, checks whether a point
            pt is contained in the area of the scan
        
        """
    def __init__(self, hdrfile, verbose = False):
        self.hdr = hdrfile
        self.label = path.basename(hdrfile[:-4])
        self.shortlbl = self.label[-3:]
        with open(hdrfile, 'r') as infile:
            ln = infile.readline()
            
            while "ScanDefinition" not in ln:
                ln = infile.readline()
            s = ln.split()
            self.scantype = ' '.join(s[s.index("Type")+2:s.index("Flags")])[1:-2]
            self.flags = ' '.join(s[s.index("Flags")+2:s.index("Dwell")])[1:-2]
            if verbose:
                print(path.basename(hdrfile) + '\t' + self.scantype)
            
            if "Focus" not in self.scantype:
                while 'Axis = { Name = \"Energy\"' not in ln:
                    ln = infile.readline()       
                ln = infile.readline()
                self.energies = ln[:ln.index(')')]
                self.energies = self.energies.split(', ')
                self.energies = [float(en) for en in self.energies[1:]]

            while "Time" not in ln:
                ln = infile.readline()
            s = ln.split(";")
            self.time = s[0][-9:-1]

            while "CentreXPos" not in ln:
                ln = infile.readline()
            s = ln.split()
            if "Line Scan" in self.scantype:
                dims = [float(s[s.index(q) + 2][:-1]) for q in ["CentreXPos", "CentreYPos", "Length", "Theta"]]
                self.xcenter = dims[0]
                self.ycenter = dims[1]
                self.xmin = dims[0] - 0.5*dims[2]*np.cos(dims[3])
                self.xmax = dims[0] + 0.5*dims[2]*np.cos(dims[3])
                self.ymin = dims[1] - 0.5*dims[2]*np.sin(dims[3])
                self.ymax = dims[1] + 0.5*dims[2]*np.sin(dims[3])
            
            elif "Focus" in self.scantype:
                dims = [float(s[s.index(q) + 2][:-1]) for q in ["CentreXPos", "CentreYPos", "Length", "Theta", "ZPStartPos", "ZPEndPos"]]
                self.xcenter = dims[0]
                self.ycenter = dims[1]
                self.xmin = dims[0] - 0.5*dims[2]*np.cos(dims[3])
                self.xmax = dims[0] + 0.5*dims[2]*np.cos(dims[3])
                self.ymin = dims[1] - 0.5*dims[2]*np.sin(dims[3])
                self.ymax = dims[1] + 0.5*dims[2]*np.sin(dims[3])
                self.zpmin = dims[4]
                self.zpmax = dims[5]
                
                ln = infile.readline()
                s = ln.split()
                self.energies = [float(s[s.index("Energy") + 2][:-1])]
            
            elif "Point" in self.scantype:
                dims = [s[s.index(q) + 2][:-1] for q in ["CentreXPos", "CentreYPos"]]
                self.xcenter = float(dims[0])
                self.ycenter = float(dims[1][:-3])
            
            else:    
                dims = [float(s[s.index(q) + 2][:-1]) for q in ["CentreXPos", "CentreYPos", "XRange", "YRange", "XStep", "YStep"]]
                self.xcenter = dims[0]
                self.ycenter = dims[1]
                self.xrange = dims[2]
                self.yrange = dims[3]
                self.xstep = dims[4]
                self.ystep = dims[5]                
                self.xmin = dims[0] - 0.5*dims[2]
                self.xmax = dims[0] + 0.5*dims[2]
                self.ymin = dims[1] - 0.5*dims[3]
                self.ymax = dims[1] + 0.5*dims[3]
                
                # Adjust dimensions for edge case when xrange or yrange is zero
                if self.xrange == 0.0:
                    self.xrange = self.ystep
                    self.xmax += self.ystep
                    self.xstep = self.ystep
                if self.yrange == 0.0:
                    self.yrange = self.xstep
                    self.ymax += self.xstep
                    self.ystep = self.xstep
                
                if "Multi-Region" in self.flags:
                    ln = infile.readline()
                    s = ln.split()
                    self.nreg = 1
                    self.regext = [[self.xmin, self.xmax, self.ymin, self.ymax]]
                    while "CentreXPos" in s:
                        self.nreg += 1
                        dims = [float(s[s.index(q) + 2][:-1]) for q in ["CentreXPos", "CentreYPos", "XRange", "YRange", "XStep", "YStep"]]
                        self.regext.append([dims[0]-0.5*dims[2], dims[0]+0.5*dims[2], dims[1]-0.5*dims[3], dims[1]+0.5*dims[3]])
                        self.xmin = min(self.xmin, self.regext[-1][0])
                        self.xmax = max(self.xmax, self.regext[-1][1])
                        self.ymin = min(self.ymin, self.regext[-1][2])
                        self.ymax = max(self.ymax, self.regext[-1][3])
                        ln = infile.readline()
                        s = ln.split()
                    self.xcenter = 0.5*(self.xmin+self.xmax)
                    self.ycenter = 0.5*(self.ymin+self.ymax)
                    self.xrange = self.xmax - self.xmin
                    self.yrange = self.ymax - self.ymin
        
        if "Point" not in self.scantype:
            ximlist = glob(hdrfile[:-4] + '*a*.xim')
            if ximlist != []:
                if "Multi-Region" in self.flags:
                    self.img = [np.loadtxt(ximlist[n])[::-1] for n in range(self.nreg)]
                else:
                    self.img = np.loadtxt(ximlist[0])[::-1]
                    if len(self.img.shape) == 1:
                        self.img = np.array([self.img])
            else:
                self.img = np.zeros((int(self.yrange/self.ystep), int(self.xrange/self.xstep)))
            
            if self.scantype == "NEXAFS Line Scan":
                egrid = np.linspace(self.energies[0], self.energies[-1], 500, endpoint = False)
                newimg = np.zeros((len(egrid), len(self.img)-1))
                
                self.img = self.img.T
                k = 0
                for j in range(len(egrid)):
                    if k < len(self.img)-1 and egrid[j] >= self.energies[k+1]:
                        k += 1
                    newimg[j] = self.img[k][:-1]
                self.img = newimg.T

    def contains(self, pt):
        return (self.xmin <= pt[0] <= self.xmax and self.ymin <= pt[1] <= self.ymax)


def scantree(sl):
    """Given a list of scans:
    
    -- Runs through the list and picks out image, line, and point scans.
    
    -- Breaks the list into sublists, based on the assumption
       that an image scan 500 microns wide or larger indicates
       a move to a new sample.
    
    -- Returns a tree in which tree[i] = [indices of scans for which
                                          at least 80% of the area
                                          (80% of the length for line
                                          scans) is within scan i] 
    """
    print('Checking scan positions...')
    
    imgslinespts = [i for i in range(len(sl)) if (sl[i].scantype == "Image Scan" or "NEXAFS" in sl[i].scantype)]
    
    subsets = []
    for j in imgslinespts:
        if (sl[j].scantype == "Image Scan" and sl[j].xrange >= 500.0) or subsets == []:
            subsets.append([])
            print("New sample starting with " + sl[j].label)
        subsets[-1].append(j)
    
    tree = [[] for i in range(len(sl))]
    
    for sub in subsets:
        for i in sub:
            if "Image" in sl[i].scantype:
                for k in sub:
                    xmin_wide = sl[i].xmin - 0.15*sl[i].xrange
                    xmax_wide = sl[i].xmax + 0.15*sl[i].xrange
                    ymin_wide = sl[i].ymin - 0.15*sl[i].yrange
                    ymax_wide = sl[i].ymax + 0.15*sl[i].yrange
                    
                    if "Point" in sl[k].scantype:
                        if sl[i].contains([sl[k].xcenter, sl[k].ycenter]):
                            tree[i].append(k)
                    
                    elif "Line" in sl[k].scantype:
                        if sl[i].contains([sl[k].xmin, sl[k].ymin]) or sl[i].contains([sl[k].xmax, sl[k].ymax]):
                            my_xmin = max(sl[i].xmin, sl[k].xmin)
                            my_xmax = min(sl[i].xmax, sl[k].xmax)
                            my_ymin = max(sl[i].ymin, sl[k].ymin)
                            my_ymax = min(sl[i].ymax, sl[k].ymax)
                            full_length = np.linalg.norm(np.array([sl[k].xmax-sl[k].xmin, sl[k].ymax-sl[k].ymin]))
                            contained_length = np.linalg.norm(np.array([my_xmax-my_xmin, my_ymax-my_ymin]))
                            if contained_length/full_length >= 0.75:
                                tree[i].append(k)
                        
                    elif "Image" in sl[k].scantype:
                        if sl[i].contains([sl[k].xmin, sl[k].ymin]) or sl[i].contains([sl[k].xmax, sl[k].ymax]) or \
                           sl[i].contains([sl[k].xmin, sl[k].ymax]) or sl[i].contains([sl[k].xmax, sl[k].ymin]):
                            full_area = sl[k].xrange * sl[k].yrange
                            # Leave out scans that are too similar in size to scan i
                            if full_area <= 0.75*(sl[i].xrange * sl[i].yrange):                                
                                my_xmin = max(sl[i].xmin, sl[k].xmin)
                                my_xmax = min(sl[i].xmax, sl[k].xmax)
                                my_ymin = max(sl[i].ymin, sl[k].ymin)
                                my_ymax = min(sl[i].ymax, sl[k].ymax)
                                contained_area = (my_xmax-my_xmin)*(my_ymax-my_ymin)
                                if contained_area >= 0.8*full_area or \
                                   (xmin_wide <= sl[k].xmin and sl[k].xmax <= xmax_wide and ymin_wide <= sl[k].ymin and sl[k].ymax <= ymax_wide):
                                    tree[i].append(k)

    # Refine the tree to keep the final
    # display from being too redundant
    for j in range(len(sl)):
        contains_j = [i for i in range(len(sl)) if j in tree[i]]
        contains_j.sort
        
        # Divide the set of scans that contain scan j
        # into the ones acquired before and after it
        prev = [i for i in contains_j if i < j]
        post = [i for i in contains_j if i > j]
        
        # Ideally, we want to display scan j on the last
        # scan before it that contains it. Otherwise,
        # for example if the user zooms out from a small
        # scan, we display scan j on the next scan after
        # it that contains it.
        reduced_contains_j = []
        if prev != []:
            reduced_contains_j.append(max(prev))
        elif post != []:
            reduced_contains_j.append(min(post))
        for i in contains_j:
            if i not in reduced_contains_j:
                tree[i] = [k for k in tree[i] if k != j]
    
    # Finally, ensure that "tree" really is
    # is a tree, i.e. if i2 contains i1 contains j,
    # make sure i2 is not listed as containing j
    for j in range(len(sl)):
        contains_j = [i1 for i1 in range(len(sl)) if j in tree[i1]]
        for i1 in contains_j:
            contains_i1 = [i2 for i2 in range(len(sl)) if i1 in tree[i2]]
            for i2 in contains_i1:
                tree[i2] = [k for k in tree[i2] if k != j]

    return tree

def gendisplay(s, p, subscans = None):
    """Takes a scan s and a matplotlib subplot p as arguments
    and fills the subplot with an image of the scan. If a list
    of subscans is also passed to the function, outlines of
    the subscans will be displayed on the image.
    
    The list of subscans follows this convention:
        subscans = [nested_once_plot, nested_once_labels, nested_twice_plot]
        nested_once_plot: each entry is a two-element list [xcoords, ycoords]
                          specifying the bounding box or line or point for the scan
        nested_once_labels: the label to be attached to the trace obtained from nested_once_plot
        nested_twice_plot: [xcoords, ycoords] for subscans of the scans in nested_once_plot
                           (no corresponding labels)
    """
    
    p.get_xaxis().get_major_formatter().set_useOffset(False)
    p.get_yaxis().get_major_formatter().set_useOffset(False)
    
    p.tick_params(axis = 'x', which = 'both', direction = 'out', labelbottom = 'off', bottom = 'off', top = 'off')
    p.tick_params(axis = 'y', which = 'both', direction = 'out', labelleft = 'off', left = 'off', right = 'off')
    
    if "Line Scan" in s.scantype:
        lg = np.linalg.norm(np.array([s.xmax-s.xmin, s.ymax-s.ymin]))
        p.imshow(s.img, cmap = cm.gray, interpolation = 'nearest',
                 extent = [s.energies[0], s.energies[-1], 0.0, lg],
                 aspect = (s.energies[-1] - s.energies[0])/lg)
        
        p.tick_params(axis = 'x', which = 'both', direction = 'out', labelbottom = 'on', bottom = 'on', top = 'off')
        p.locator_params(axis = 'x', tight = True, nbins = 6)
        p.locator_params(axis = 'y', nbins = 6)
        p.set_xlabel('Energy (eV)', labelpad = 2)
    
    elif "Focus" in s.scantype:
        lg = np.linalg.norm(np.array([s.xmax-s.xmin, s.ymax-s.ymin]))
        p.imshow(s.img, cmap = cm.gray, interpolation = 'nearest',
                 extent = [0.0, lg, s.zpmin, s.zpmax], aspect = lg/(s.zpmax-s.zpmin))
        p.locator_params(nbins = 5)
        p.tick_params(axis = 'y', which = 'both', direction = 'out', labelleft = 'on', left = 'on', right = 'off')
        p.set_ylabel('Zone plate position ($\mathregular{\\mu}$m)', labelpad = 1)
    
    else:
        if "Multi-Region" in s.flags:
            for n in range(s.nreg):
                p.imshow(s.img[n], cmap = cm.gray, interpolation = 'nearest', extent = s.regext[n])
            p.set_xlim(s.xmin, s.xmax)
            p.set_ylim(s.ymin, s.ymax)
        else:
            p.imshow(s.img, cmap = cm.gray, interpolation = 'nearest',
                     extent = [s.xmin, s.xmax, s.ymin, s.ymax])
    
    ex = p.axis()
    a = p.get_aspect()
    if (a == 'equal' or "Image" in s.scantype) and ex[3]-ex[2] < ex[1]-ex[0]:
        box = p.get_position()
        p.set_position([box.x0, box.y0 + 0.5*(s.xrange - s.yrange)/s.xrange*box.height, box.width, box.height])
    if (a == 'equal' or "Image" in s.scantype) and ex[1]-ex[0] < ex[3]-ex[2]:
        box = p.get_position()
        p.set_position([box.x0 + 0.5*(s.xrange - s.yrange)/s.yrange*box.width, box.y0, box.width, box.height])
        rightedge = 1.0 + (s.yrange - s.xrange)/s.xrange
    else:
        rightedge = 1.0
    
    p.text(0.0, 1.02, s.label + '\n' + s.scantype, horizontalalignment = 'left', verticalalignment = 'bottom', transform = p.transAxes)
    rightcol = s.time + '\n'
    if 'NEXAFS' not in s.scantype:
        rightcol += str(np.around(s.energies[0], decimals = 1)) + ' eV'
    elif s.scantype == 'NEXAFS Image Scan':
        ximlist = glob(s.hdr[:-4] + '*a*.xim')
        if "Multi-Region" in s.flags:
            last_energy = s.energies[int(len(ximlist)/s.nreg)-1]
        else:
            last_energy = s.energies[len(ximlist)-1]
        lowercaption = str(np.around(s.energies[0], decimals = 1)) + ' eV' + ' - ' + \
                       str(np.around(last_energy, decimals = 1)) + ' eV\n' + str(len(ximlist))
        if len(ximlist) == 1:
            lowercaption += ' image'
        else:
            lowercaption += ' images'
        if len(ximlist) < len(s.energies):
            lowercaption += ' (aborted)'
        p.text(0.0, -0.04, lowercaption, horizontalalignment = 'left', verticalalignment = 'top', transform = p.transAxes)
    p.text(rightedge, 1.02, rightcol, horizontalalignment = 'right', verticalalignment = 'bottom', transform = p.transAxes)
    
    
    # Scale bar
    if s.scantype in ["Image Scan", "NEXAFS Image Scan", "OSA Scan", "Detector Scan"]:
        scale = max([x for x in [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000] if x <= 0.2*(s.xmax - s.xmin)])
        # Check if image is mostly dark or light in lower left
        if "Multi-Region" in s.flags:
            scalecolor = '#000000'
        else:
            corner = s.img[int(0.8*len(s.img)):,:int(0.3*len(s.img[0]))]
            m = np.mean(corner)
            if abs(m - np.max(s.img)) < abs(m - np.min(s.img)):
                scalecolor = '#222222'
            else:
                scalecolor = '#dddddd'
        
        p.plot([s.xmin + 0.05*s.xrange, s.xmin + 0.05*s.xrange + scale],
             [s.ymin + 0.05*s.xrange, s.ymin + 0.05*s.xrange],
             color = scalecolor, linewidth = 3.0, scalex = False, scaley = False)
        p.text(s.xmin + 0.03*s.xrange, s.ymin + 0.08*s.xrange,
             str(scale) + ' $\mathregular{\mu}$m', horizontalalignment = 'left', color = scalecolor)
    
    # Contained scans
    if "Image" in s.scantype and subscans is not None:
        p.autoscale(False)
        labels = []
        labelcoords = []
        
        # Cluster image scans by how close their centers are
        imgcoords = [subscans[0][i] for i in range(len(subscans[1])) if len(subscans[0][i][0]) == 5]
        imglbls = [subscans[1][i] for i in range(len(subscans[1])) if len(subscans[0][i][0]) == 5]
        centers = [np.array([0.5*(su[0][0]+su[0][1]), 0.5*(su[1][1]+su[1][2])]) for su in imgcoords]
        areas = [(su[0][1]-su[0][0])*(su[1][2]-su[1][1]) for su in imgcoords]
        widths = [su[0][1]-su[0][0] for su in imgcoords]
        heights = [su[1][2]-su[1][1] for su in imgcoords]
        bins = [0 for n in range(len(imgcoords))]
        b = 0
        for k in range(len(centers)):
            if bins[k] == 0:
                distances = [np.linalg.norm(centers[k] - c) for c in centers]
                rebin = []
                for q in range(len(distances)):
                    if distances[q] <= 0.25*min(s.xrange, s.yrange) and 0.7*areas[k] <= areas[q] <= 1.3*areas[k]:
                        dx = widths[q]
                        dy = heights[q]
                        if q == k or max(dx, dy) >= 0.2*min(s.xrange, s.yrange) or \
                           0.9*widths[k] <= widths[q] <= 1.1*widths[k] or \
                           0.9*heights[k] <= heights[q] <= 1.1*heights[k] or \
                           ((imgcoords[k][0][0] <= imgcoords[q][0][0] <= imgcoords[k][0][1] or imgcoords[k][0][0] <= imgcoords[q][0][1] <= imgcoords[k][0][1]) and \
                           (imgcoords[k][1][1] <= imgcoords[q][1][1] <= imgcoords[k][1][2] or imgcoords[k][1][1] <= imgcoords[q][1][2] <= imgcoords[k][1][2])):
                            rebin.append(q)
                
                newb = max([bins[q] for q in rebin])
                if newb == 0:
                    b += 1
                    newb = b
                for q in rebin:
                    bins[q] = newb
                        
        if bins != []:
            binvals = []
            for b in bins:
                if b not in binvals:
                    binvals.append(b)
            for k in range(len(binvals)):
                b = binvals[k]
                bset = [i for i in range(len(bins)) if bins[i] == b]
                bset.sort()
                my_xmin = min([imgcoords[i][0][0] for i in bset])
                my_xmax = max([imgcoords[i][0][1] for i in bset])
                my_ymin = min([imgcoords[i][1][1] for i in bset])
                my_ymax = max([imgcoords[i][1][2] for i in bset])
                
                labelcoords.append([my_xmin, my_xmax, my_ymin, my_ymax])
                if len(bset) > 2:
                    steps = [0] + [int(imglbls[bset[j]]) - int(imglbls[bset[j-1]]) for j in range(1,len(bset))]
                    if max(steps) == 1:
                        labels.append(imglbls[bset[0]] + '-' + imglbls[bset[-1]])
                    else:
                        longlbl = []
                        breaklist = [j for j in range(len(steps)) if steps[j] != 1]
                        for q in range(len(breaklist)-1):
                            if breaklist[q+1] == breaklist[q]+1:
                                longlbl.append(imglbls[bset[breaklist[q]]])
                            else:
                                longlbl.append(imglbls[bset[breaklist[q]]] + '-' + imglbls[bset[breaklist[q+1]-1]])
                        if breaklist[-1] == len(bset)-1:
                            longlbl.append(imglbls[bset[breaklist[-1]]])
                        else:
                            longlbl.append(imglbls[bset[breaklist[-1]]] + '-' + imglbls[bset[-1]])
                        labels.append(', '.join(longlbl))
                else:
                    labels.append(', '.join([imglbls[i] for i in bset]))
            
                for i in bset:
                    plt.plot(*imgcoords[i], color = 'red')
        
        
        for i in range(len(subscans[0])):
            if len(subscans[0][i][0]) != 5:
                coords = subscans[0][i]
                lbl = subscans[1][i]
                if len(coords[1]) == 1:
                    # Point scan
                    plt.scatter(*coords, color = 'red', s = 3.0)
                    labelcoords.append([coords[0][0], coords[0][0], coords[1][0], coords[1][0]])
                    labels.append(lbl)
                else:
                    # Line scans
                    plt.plot(*coords, color = 'red')
                    labelcoords.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])
                    labels.append(lbl)
        
        ## Sub-sub scans
        # for i in range(len(subscans[2])):
        #     coords = subscans[2][i]
        #     if len(coords[1]) == 1:
        #         # Point scan
        #         plt.scatter(*coords, color = 'red')
        #     else:
        #         # Line scans and Image Scans
        #         plt.plot(*coords, color = 'red')
        
        
        # Place labels
        dl = 0.005*max(s.xrange, s.yrange)
        bdy = 0.1*max(s.xrange, s.yrange)
        for i in range(len(labelcoords)):
            if labelcoords[i][0] == labelcoords[i][1] and labelcoords[i][2] == labelcoords[i][3]:
                # Point scan
                if s.xmin + bdy <= labelcoords[i][0] <= s.xmax - bdy:
                    labelx = labelcoords[i][0]
                    haln = 'center'
                elif s.xmin + bdy > labelcoords[i][0]:
                    labelx = labelcoords[i][0] + dl
                    haln = 'left'
                else:
                    labelx = labelcoords[i][0] - dl
                    haln = 'right'
                if s.ymin + 3.0*bdy >= labelcoords[i][2]:
                    labely = labelcoords[i][2] + 2.5*dl
                    valn = 'bottom'
                else:
                    labely = labelcoords[i][2] - 2.5*dl
                    valn = 'top'
            
            elif labelcoords[i][0] == labelcoords[i][1] or labelcoords[i][2] == labelcoords[i][3]:
                # Line scan
                if s.xmin + bdy <= labelcoords[i][0]:
                    labelx = labelcoords[i][0] - 2.0*dl
                    haln = 'right'
                    labely = labelcoords[i][2]
                    valn = 'center'
                elif s.xmax - bdy >= labelcoords[i][0]:
                    labelx = labelcoords[i][1] + dl
                    haln = 'left'
                    labely = labelcoords[i][2]
                    valn = 'center'
                else:
                    labelx = s.xmin + dl
                    haln = 'left'
                    labely = labelcoords[i][2] + dl
                    valn = 'bottom'
            
            else:
                # Image scan
                if s.xmin + bdy >= labelcoords[i][0] or s.ymin + 3.0*bdy >= labelcoords[i][2]:
                    labelx = min(s.xmax, labelcoords[i][1])
                    haln = 'right'
                else:
                    labelx = labelcoords[i][0]
                    haln = 'left'
                if s.ymax - bdy <= labelcoords[i][3]:
                    labely = labelcoords[i][2] - dl
                    valn = 'top'
                else:
                    labely = labelcoords[i][3]
                    valn = 'bottom'
            
            p.text(labelx, labely, labels[i], color = 'red', horizontalalignment = haln, verticalalignment = valn)
        # bboximage = np.ones_like(s.img)
        # xstep = s.xrange/len(s.img[0])
        # ystep = s.yrange/len(s.img[1])
        # for i in range(len(labelcoords)):
        #     x_ind_min = 
            
    
def genspectrum(s, p, flist):
    """Takes a scan class (s), a subplot (p), and a
    list of spectrum files (flist) as arguments.
    Fills subplot p with a plot of the spectra."""
    
    for f in flist:
        with open(f, 'r') as infile:
            tmp = infile.readlines()
        tmp = [ln for ln in tmp if ('#' not in ln and '%' not in ln)]
        tmp = [ln.split() for ln in tmp]
        tmp = np.array(tmp).astype(np.float)
        tmp = tmp.T
        
        if "Point" in s.scantype:
            if np.max(tmp[1]) > 0:
                pow10 = int(np.log10(np.max(tmp[1])))
                scale = 10**pow10
                tmp[1] /= scale
            else:
                pow10 = 0
        
        p.locator_params(axis = 'x', tight = True, nbins = 6)
        p.plot(tmp[0], tmp[1], label = path.basename(f))
    
    if len(flist) > 1:
        p.legend(loc = 4, fontsize = 6.0)

    p.get_xaxis().get_major_formatter().set_useOffset(False)
    p.get_yaxis().get_major_formatter().set_useOffset(False)
    
    p.set_xlabel('Energy (eV)', labelpad = 1)
    if "Point" in s.scantype:
        p.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,3))
        p.set_ylabel('Counts / $\\mathregular{10^{' + str(pow10) + '}}$', labelpad = 1)

    p.text(0.0, 1.02, s.label + '\n' + s.scantype, horizontalalignment = 'left', verticalalignment = 'bottom', transform = p.transAxes)
    p.text(1.0, 1.02, s.time + '\n', horizontalalignment = 'right', verticalalignment = 'bottom', transform = p.transAxes)

def buildpdf(sl, tr, wdir, outdir = None, blname = ''):
    """Inputs: list of scans sl
               dictionary tr indicating which scans are nested
       
       Generates the final pdf output."""
    
    panellist = []
    
    specfiles = glob(path.join(wdir, '*.txt')) + glob(path.join(wdir, '*', '*.txt'))
    specfiles = [f for f in specfiles if 'i0' not in f]
    specheaders = []
    for f in specfiles:
        with open(f, 'r') as infile:
            lns = infile.readlines()
        lns = [r for r in lns if '#' in r or '%' in r]
        specheaders.append(''.join(lns))
    
    for i in range(len(sl)):
        panellist.append([i])
        if sl[i].scantype == 'NEXAFS Image Scan' or sl[i].scantype == 'NEXAFS Line Scan':
            check_specfiles = [k for k in range(len(specheaders)) if sl[i].label in specheaders[k]]
            for k in check_specfiles:
                print('Found spectrum file derived from ' + path.basename(sl[i].hdr) + ': ' + path.basename(specfiles[k]) )
            if check_specfiles != []:
                panellist.append([i, [specfiles[k] for k in check_specfiles]])
    
    npages = int(len(panellist)/12)
    if len(panellist)%12 != 0:
        npages += 1
    npages = str(npages)
    
    print('Building logbook file... ')
    if outdir is None:
        my_outdir = wdir
    else:
        my_outdir = outdir
    with PdfPages(path.join(my_outdir, path.basename(wdir) + '_log.pdf')) as pdf:
    
        for k in range(0, len(panellist), 12):
            page = plt.figure(figsize = (8.5,11.0))
            
            if k == 0:
                my_date = path.basename(wdir)
                my_date = '20' + my_date[:2] + '/' + my_date[2:4] + '/' + my_date[4:]
                page.text(0.093, 0.975, blname + ' logbook file for ' + my_date, horizontalalignment = 'left', size = 12, weight = 'bold')
                page.text(0.970, 0.975, 'Generated ' + strftime('%Y/%m/%d at %I:%M %p', localtime()), horizontalalignment = 'right', size = 12)
            
            for j in range(k, min(k+12, len(panellist))):
                panel = page.add_subplot(4, 3, j-k+1)
                box = panel.get_position()
                dx = 0.27*(((j-k)%3)-0.5)*box.width
                panel.set_position([box.x0 + dx, box.y0 + 0.05-0.03*int((j-k)/3), box.width*0.9, box.height*0.9])
                
                my_scan = sl[panellist[j][0]]
                if "Point" in my_scan.scantype:
                    genspectrum(my_scan, panel, glob(my_scan.hdr[:-4] + '*.xsp'))
                elif len(panellist[j]) > 1:
                    genspectrum(my_scan, panel, panellist[j][1])
                else:
                    # Build lists of data on contained scans
                    nested_once = tr[panellist[j][0]]
                    
                    nested_once_plot = []
                    nested_once_labels = []
                    nested_twice_plot = []
                    for n in nested_once:
                        subscan = sl[n]
                        nested_once_labels.append(subscan.shortlbl)
                        if "Point" in subscan.scantype:
                            nested_once_plot.append([[subscan.xcenter],[subscan.ycenter]])
                        elif "Line" in subscan.scantype:
                            nested_once_plot.append([[subscan.xmin, subscan.xmax],[subscan.ymin, subscan.ymax]])
                        else:
                            nested_once_plot.append([[subscan.xmin, subscan.xmax, subscan.xmax, subscan.xmin, subscan.xmin],
                                                     [subscan.ymin, subscan.ymin, subscan.ymax, subscan.ymax, subscan.ymin]])
                        
                        for q in tr[n]:
                            subscan = sl[q]
                            if "Point" in subscan.scantype:
                                nested_twice_plot.append([[subscan.xcenter],[subscan.ycenter]])
                            elif "Line" in subscan.scantype:
                                nested_twice_plot.append([[subscan.xmin, subscan.xmax],[subscan.ymin, subscan.ymax]])
                            else:
                                nested_twice_plot.append([[subscan.xmin, subscan.xmax, subscan.xmax, subscan.xmin, subscan.xmin],
                                                         [subscan.ymin, subscan.ymin, subscan.ymax, subscan.ymax, subscan.ymin]])
                        
                    gendisplay(my_scan, panel, subscans = [nested_once_plot, nested_once_labels, nested_twice_plot])
            
            pdf.savefig()
            print('Page ' + str(int(k/12)+1) + '/' + npages + ' done')
            page.clear()


if __name__ == '__main__':
    with open('STXM_log_config.txt') as cfgfile:
        cfg = cfgfile.readlines()
    cfg = [s[:s.index('#')] for s in cfg]
    cfg = [' '.join(s.split()) for s in cfg]
    
    datadir = cfg[0]
    logdir = cfg[1]
    bl = cfg[2]
    
    matplotlib.rcParams.update({'font.size': 9.5, 'axes.titlesize': 11.0, 'font.family': 'Arial'})

    if len(argv) > 1:
        workdir = path.abspath(argv[1])
    else:
        workdir = path.join(datadir, (datetime.date.today()).strftime('%y%m%d'))
        
        if datetime.datetime.now().hour < 2:
            yesterday = datetime.date.today() - datetime.timedelta(1)
            print('Checking whether logfile for ' + yesterday.strftime('%y%m%d') + ' is complete...')
            prevdir = path.join("Z:\\", yesterday.strftime('%y%m%d'))
            logflist = glob(path.join(logdir, yesterday.strftime('%y%m%d') + '_log.pdf'))
            if logflist == [] or datetime.datetime.fromtimestamp(path.getmtime(logflist[0])).date() != datetime.date.today():
                hdrlist = glob(path.join(prevdir, '*.hdr')) + glob(path.join(prevdir, '*', '*.hdr'))
                if hdrlist == []:
                    print('No data found for ' + yesterday.strftime('%y%m%d') + '!')
                else:
                    hdrlist = sorted(hdrlist, key = lambda f: path.basename(f))
                    scanlist = [scan(h, verbose = True) for h in hdrlist]
                    
                    datatree = scantree(scanlist)
                
                    buildpdf(scanlist, datatree, prevdir, logdir, bl)
    
    hdrlist = glob(path.join(workdir, '*.hdr')) + glob(path.join(workdir, '*', '*.hdr'))
    if hdrlist == []:
        print('No data found for ' + path.basename(workdir) + '!')
    else:
        hdrlist = sorted(hdrlist, key = lambda f: path.basename(f))
        scanlist = [scan(h, verbose = True) for h in hdrlist]
        
        datatree = scantree(scanlist)
        
        buildpdf(scanlist, datatree, workdir, logdir, bl)