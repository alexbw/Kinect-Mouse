import numpy as np
from pylab import *

def calcPerimeter(polygon):
    x,y = polygon
    numPoints = len(x)
    assert len(x) == len(y), "X and Y arrays must have the same number of points"
    perimeter = 0
    for i in range(numPoints-1):
        perimeter += np.sqrt( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )
    perimeter += np.sqrt( (x[0] - x[-1])**2 + (y[0] - y[-1])**2 )
    return perimeter

def calcArea(polygon):
    x,y = polygon
    area = 0.0
    for i in np.arange(len(x))-1:
        area += (x[i]*y[i+1] - x[i+1]*y[i])

    return abs(area)*0.5

def calcPosition(polygon):
    x,y = polygon
    return np.mean(x), np.mean(y)

def calcVelocity(positions, timestamps, return_angle=False):
    """
    positions - list of (x,y) tuples
    timestamps - the timestamp of each (x,y) tuple
    """
    numPoints = len(positions)
    assert numPoints == len(timestamps), "The number of (x,y) positions and timestamps must match"
    velocity_magnitudes = np.zeros((numPoints,))
    if return_angle is True:
        velocity_angles = np.zeros_like(velocity_magnitudes)
    for i in range(len(positions)-1):
        x1,y1 = positions[i]
        x2,y2 = positions[i+1]
        dt = timestamps[i+1] - timestamps[i]
        velocity_magnitudes[i] = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )/dt
        if return_angle is True:
            velocity_angles[i] = np.arctan2( (y2 - y1), (x2 - x1) )
        
    if return_angle is True:
        return velocity_magnitudes, velocity_angles
    else:
        return velocity_magnitudes
        
def calcTailPosition(splineFit):
    """
    INPUT:
    splineFit: 2xN array of (x,y) points fitting a contour of an animal.
    
    OUTPUT:
    tail_position, as a single (x,y) tuple
    """
    return tail_position

def zscore(data):
    """
    Normalize data column-wise to 0 mean unitary standard deviation
    """
    means = data.mean(0)
    stds = np.std(data, 0)
    data = (data - means) / stds
    data[np.isnan(data)] = 0.0
    
    return data

def flattenExperimentStructArray(structArray):
    return structArray.view('%df4' % len(np.hstack(structArray[0])))
    
def concatenate_data(fits, ellipses, imgs, diffimgs, frame_ids, normalize=True, flatten=True):
    """
    maxHeight
    medianHeight
    aspectRatio
    perimeter
    area
    velocity
    accel
    dAngle
    allImgData
    allDiffImgData
    """
    
    # Calculate all of the different properties listed above
    maxHeights = np.asarray( [i.max() for i in imgs] )
    meanHeights = np.asarray( [np.mean(i) for i in imgs] )
    perimiters = np.asarray( [calcPerimeter(fit) for fit in fits] )
    areas = np.asarray( [calcArea(fit) for fit in fits] )
    
    positions = np.asarray( [calcPosition(fit) for fit in fits] )
    velocities = calcVelocity(positions, frame_ids)
    angles = np.asarray( [e[0][2] for e in ellipses] )
    dAngles = np.r_[np.diff(angles), 0]
    
    # Define a dtype that'll make accessing dimensions easier
    exp_dtype = np.dtype([('maxheight', 'f4'), 
                                ('meanheight', 'f4'), 
                                ('perimeter', 'f4'), 
                                ('area', 'f4'), 
                                ('velocity', 'f4'),
                                ('diffangles', 'f4'),
                            ('img', '%df4' % imgs1[0].size),
                            ('diffimg', '%df4' % diffimgs1[0].size)])

    
    # Stick everything all up ins boyyy
    data = np.zeros((len(imgs),), dtype=exp_dtype)
    data['maxheight'] = maxHeights
    data['meanheight'] = meanHeights
    data['perimeter'] = perimiters
    data['area'] = areas
    data['velocity'] = velocities
    data['diffangles'] = dAngles
    data['img'] = np.vstack( [img.ravel() for img in imgs] )
    data['diffimg'] = np.vstack( [diffimg.ravel() for diffimg in diffimgs] )
    
    # TODO:
    # Calculate windowed frequency and cepstral coefficients (start with just a single window size)
    
    # Normalize over each dimension
    if normalize:
        data = zscore(flattenExperimentStructArray(data))
    if not flatten:
        data.dtype = exp_dtype
    
    return data
    
# Number-crunching functions
def dimensionally_reduce(data, output_dim = 6, pcaNode=None):
    import mdp
       
    # Spit those out
    if pcaNode == None:
        pcaNode = mdp.nodes.PCANode(output_dim = output_dim)
        pcaNode.train(data)
        
    princomp = pcaNode.execute(data)

    return princomp, pcaNode

def cluster_data(data, num_clusters = 10, km=None):
    from sklearn import cluster
    if km is None:
        km = cluster.KMeans(init='k-means++', k=num_clusters, n_init=100)
        km.fit(data)
    labels = km.predict(data)
    
    return labels, km

# Display Functions
def show_pca(princomp, i1=0, i2=1):
    plot(princomp[:,i1], princomp[:,i2], '.')
    
def show_scatterplot_heatmap(princomp, i1=0, i2=1, bins=(50,50), binrange=None):

    x = princomp[:,i1].astype('float32')
    y = princomp[:,i2].astype('float32')
    
    if binrange == None:
        binrange = ((x.min(), x.max()), (y.min(), y.max()))
    assert(len(x) == len(y)), "X, Y an data must be the same length"

    # Normalize the values of x and y, (it'll make it easier to index later)
    x = np.clip(x, binrange[0][0], binrange[0][1])
    x -= binrange[0][0]
    x /= binrange[0][1] - binrange[0][0]
    x *= bins[0]-1

    y = np.clip(y, binrange[1][0], binrange[1][1])
    y -= binrange[1][0]
    y /= binrange[1][1] - binrange[1][0]
    y *= bins[1]-1

    density = np.ones((bins[0], bins[1]))
    for i in np.arange(len(x)):
        density[x[i],y[i]] += 1

    imshow(np.sqrt(density.T), origin='lower', interpolation='bilinear')
    
    return density, binrange

def show_each_cluster(data, labels,i1=0,i2=1, x_range=(0,450), y_range=(0,500), color='r'):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    rows = 4
    columns = 2
    
    for label, i in zip(unique(labels), range(num_clusters)):
        subplot(rows, columns, i+1)
        idx = np.where(labels == label)[0]
        plot(data[idx,i1], data[idx,i2], marker=".", linewidth=0, color=color)
        xlim(x_range[0], x_range[1])
        ylim(y_range[0], y_range[1])

def show_clustered_data(data, labels, i1=0, i2=1, colormap = cm.jet):
    unique_labels = np.unique(labels)
    
    lookup = np.choose(unique_labels, np.linspace(0,0.8,len(unique_labels)))
    colors = colormap(lookup)
    
    for color, label in zip(colors,unique_labels):
        idx = np.where(labels == label)[0]
        plot(data[idx,i1], data[idx,i2], marker = ".", linewidth=0, color=color)


def show_image_princomp(pcaNode, which_comp=0):
    principal_component = pcaNode.get_projmatrix()[6:,:].reshape((20,80,num_dimensions))
    figure(figsize=(6,8))
    imshow(q[:,:,which_comp])
    title("Principal component %d" % which_comp)
    colorbar()

def calcMeanImgsInClusters(labels, imgs):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    mean_imgs = [0]*num_clusters
    for i in unique_labels:
        this_idx = np.where(labels == i)[0]
        mean_imgs[i] = np.mean([imgs[j] for j in this_idx], 0) / np.sqrt(len(this_idx))
        
    return mean_imgs

def calcNormMeanImgsInClusters(labels, imgs):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    mean_imgs = [0]*num_clusters
    std_imgs = [0]*num_clusters
    
    
    for i in unique_labels:
        this_idx = np.where(labels == i)[0]
        
        mean_imgs[i] = np.median([imgs[j] for j in this_idx], 0)
        std_imgs[i] = np.std([imgs[j] for j in this_idx], 0) / np.sqrt(len(this_idx))
        
    return mean_imgs, std_imgs


def calcMeanCombinedImgsInClusters(labels, imgs, diffimgs):
    meanimgs = calcMeanImgsInClusters(labels, imgs)
    meandiffimgs = calcMeanImgsInClusters(labels, diffimgs)
    
    combinedimgs = []
    for i in range(len(meanimgs)):
        # NOTE NOTE NOTE:
        # SCALING THE DIFFERENCE IMAGE SO THAT IT IS MORE VISIBLE
        combinedimgs.append( np.hstack((meanimgs[i], 3*meandiffimgs[i])) )
        
    return combinedimgs

def showMeanCombinedImgsInClusters(labels, imgs, diffimgs, colorlimits=(-1,12)):
    mean_imgs = calcMeanCombinedImgsInClusters(labels, imgs, diffimgs)
    unique_labels = np.unique(labels)
    numRows = 4
    figure(figsize=(4*num_clusters/numRows, 2*numRows))
    for i in unique_labels:
        subplot(numRows, num_clusters/numRows, i)
        imshow(mean_imgs[i])
        clim(colorlimits[0], colorlimits[1])
        
def show_annotated_aligned_movie(frames, labels, framerate=30):
    """
    Okay, it's about time that we showed mouse_images. It is seriously about time.
    """
    from time import sleep
    import SimpleCV as scv
    assert len(frames) == len(labels), "Must have equal numbers of images and labels"
    assert frames[0].has_key('mouse_image'), "We require the frames to have the mouse_image key"
    
    num_frames = len(frames)
    
    for i in range(num_frames):
        img = scv.Image(frames[i]['mouse_image']*2.0)
        img = img.scale(4.0)
        img.drawText(str(labels[i]), x=5, y=5, fontsize=40, color=(255,255,255))
        img.show()
        sleep(1.0 / framerate)
  
def show_annotated_movie(frames, labels, full_size=(240,240), framerate=30, text_offset=5):  
    """
    This takes the mouse's image and embeds it into a larger image.
    """
    from time import sleep
    import SimpleCV as scv
    import scipy.ndimage.interpolation as interp
    
    assert len(frames) == len(labels), "Must have equal numbers of images and labels"
    assert frames[0].has_key('mouse_image'), "We require the frames to have the mouse_image key"
    assert frames[0].has_key('centroid_x'), "We require the frames to have the centroid_x key"
    assert frames[0].has_key('centroid_y'), "We require the frames to have the centroid_y key"
    assert frames[0].has_key('angle'), "We require the frames to have the angle key"
    
    num_frames = len(frames)
    
    for i in range(num_frames):
        # Get the raw image
        raw_img = frames[i]['mouse_image']*2.0
        centroid_x = int(frames[i]['centroid_x'])
        centroid_y = int(frames[i]['centroid_y'])
        
        # Rotate it
        raw_img = interp.rotate(raw_img, frames[i]['angle'], mode='constant')
                
        # Now, we can actually place it inside the image
        
        new_img = np.zeros(full_size)

        idx_h = np.r_[0:raw_img.shape[0]] - raw_img.shape[0]/2 + centroid_y
        idx_w = np.r_[0:raw_img.shape[1]] - raw_img.shape[1]/2 + centroid_x
        
        lh = centroid_y - raw_img.shape[0]/2
        rh = centroid_y + raw_img.shape[0]/2
        lw = centroid_x - raw_img.shape[1]/2
        rw = centroid_x + raw_img.shape[1]/2
        
        lh = max(0,lh)
        rh = min(full_size[0]-1,rh)
        lw = max(0,lw)
        rw = min(full_size[1]-1,rw)        
        
        lh_img = 0
        rh_img = rh - lh
        lw_img = 0
        rw_img = rw - lw
        
        new_img[lh:rh,lw:rw] = raw_img[lh_img:rh_img,lw_img:rw_img]
        
        img = scv.Image(new_img)
        img.drawText(str(labels[i]), x=centroid_y+text_offset, y=centroid_x+text_offset, fontsize=25, color=(255,255,255))
        img.show()
        sleep(1.0 / framerate)
        
def smooth_channels(signal, std=2):
    """
    Smooths a numpoints x numchannels array
    """
    smoothed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        smoothed[:,i] = smooth(signal[:,i], std)
        
    return smoothed
        
def smooth(signal, std=2):
    "Smooths a 1D signal"
    from scipy.signal import convolve, boxcar, gaussian
    smoothingKernel = gaussian(std*5,std)
    smoothingKernel /= np.sum(smoothingKernel)
    
    signal = np.convolve(signal, smoothingKernel, 'same')
    
    return signal
    

def show_spines_in_clusters2(spines, labels, velocities=None, xbounds=(0,80), ybounds=(10,75)):
    """
    Modified from show_spines_in_clusters to not use subplots (offseting line plots in one axis)
    """
    
    cluster_contour = []
    diff_cluster_contour = []
    diff_spines = np.diff(spines,1,0)
    for i in np.unique(labels):
        cluster_contour.append(np.mean(spines[labels==i,:],0))
        diff_cluster_contour.append( np.mean( diff_spines[labels[:-1]==i,:],0 ) )
    cluster_contour = np.vstack(cluster_contour)
    diff_cluster_contour = np.vstack(diff_cluster_contour)
            
    if velocities == None:
        velocities = np.zeros(len(np.unique(labels)))
    
    numRows = 4
    numColumns = ceil(len(cluster_contour)/float(numRows))
    hold('on')
    
    for i in range(len(cluster_contour)):
        this_contour = cluster_contour[i,:]
        this_diff_contour = diff_cluster_contour[i,:]
        offset = i*(ybounds[1])

        x = np.arange(len(this_contour))
        plot(x, this_contour+offset, '-k', linewidth=min_width*(1+40*cluster_size[i]))
        plot(x+velocities[i], this_contour+this_diff_contour*2 + offset, '-r')

    axis('off');
    ylim(ybounds[0]-5, ybounds[1]*len(cluster_contour));
    xlim(xbounds[0], xbounds[1])

        
def show_spines_in_clusters(spines, labels, velocities=None, xbounds=(0,80), ybounds=(10,75), showerror=False, showmembership=False, color='r', numRows=1):

    cluster_contour = []
    error_cluster_contour = []
    diff_cluster_contour = []
    diff_spines = np.diff(spines,1,0)
    for i in np.unique(labels):
        idx = labels==i
        n = idx.sum()
        idx = np.where(idx)[0]
        cluster_contour.append(np.mean(spines[idx,:],0))
        error_cluster_contour.append(np.std(spines[idx,:],0) / np.sqrt(n))
        diff_cluster_contour.append( np.mean( diff_spines[labels[:-1]==i,:],0 ) )
    cluster_contour = np.vstack(cluster_contour)
    diff_cluster_contour = np.vstack(diff_cluster_contour)
    error_cluster_contour = np.vstack(error_cluster_contour)
    
    num_points = float(len(labels))
    cluster_size = []
    for i in np.unique(labels):
        proportion = (labels==i).sum() / num_points
        cluster_size.append(proportion)
        
    if velocities == None:
        velocities = np.zeros_like(cluster_size)
    
    min_width = 1
    

    numColumns = ceil(len(cluster_contour)/float(numRows))
    for i in range(len(cluster_contour)):
        this_contour = cluster_contour[i,:]
        this_diff_contour = diff_cluster_contour[i,:]
        this_error_contour = error_cluster_contour[i,:]*2
        subplot(numRows, numColumns, i)
        x = np.arange(len(this_contour))
        if showmembership:
            plot(x, this_contour, '-k', linewidth=min_width*(1+40*cluster_size[i]))
        else:
            plot(x, this_contour, '-k')
            
        if showerror == False:
            plot(x+velocities[i], this_contour+this_diff_contour*2, '-r')
        else:
            fill_between(x, this_contour-this_error_contour, this_contour+this_error_contour, alpha = 0.5, color=color)
        axis('off');
        ylim(ybounds[0], ybounds[1]);
        xlim(xbounds[0], xbounds[1])

def stacked_plot(signals, colors=None, normalize=False, colormap=cm.jet):
    """
    signals is a numpoints x numchannels 2D array.
    We're going to plot the proportion of the data in each timeslice. 
    """
        
    # Normalize, if we want.
    if normalize:
        divisor = np.atleast_2d(signals.sum(1)).T.repeat(signals.shape[1],1)
        signals = signals / divisor
    
    to_plot = np.cumsum(signals,1)
    num_pts = to_plot.shape[0]
    to_plot = np.hstack((np.zeros((num_pts,1)), to_plot))
    
    num_sections = to_plot.shape[1]-1
    if colors == None:
        colors = colormap(np.linspace(0,1,num_sections+1))[:-1]
    
    for i in range(num_sections):
        fill_between(range(num_pts), to_plot[:,i], to_plot[:,i+1], color=colors[i,:], linewidth=0)
    axis('tight')
    yticks([])
    xlabel("Frame #")

def calc_qbps(fits, ellipses, imgs, diffimgs, frame_ids, num_dimensions=6, num_qbps=8, pcaNode=None, kmeansData=None):
    data = concatenate_data(fits, ellipses, imgs, diffimgs, frame_ids)
    
    princomp, pcaNode = dimensionally_reduce(data, num_dimensions, pcaNode)
    qbp_ids, kmeansData = cluster_data(princomp, num_qbps, kmeansData)

    return princomp, qbp_ids, pcaNode, kmeansData
    
    
def rotateAndCrop(img, angle, origSize=(80,40), downSize=(40, 20)):
	import Image
	rotatedImg = Image.fromarray(img).rotate(-angle, expand=True)
	
	# Crop the PIL image to a standard frame, then downsize it
	xCenter, yCenter = rotatedImg.size
	xCenter /= 2
	yCenter /= 2
	cropRegion = (xCenter-origSize[0]/2, 
				  yCenter-origSize[1]/2, 
				  xCenter+origSize[0]/2, 
				  yCenter+origSize[1]/2)
	rotatedImg = rotatedImg.crop(cropRegion).resize(downSize)

	return np.asarray(rotatedImg)
