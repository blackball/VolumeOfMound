#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @blackball (bugway@gmail.com)

import open3d as o3d
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPoint

def TriVolume(t):
    """
    https://www.mathpages.com/home/kmath393.htm#:~:text=So%2C%20if%20we%20let%20A1,x4y2)%5D%2F2%20and%20of
    """
    t = t[np.argsort(t[:, 2])] # sort by z
    x1,y1,z1 = t[0]
    x2,y2,z2 = t[1]
    x3,y3,z3 = t[2]
    if z1 < 0 or z1 < 0 or z3 < 0:
        return 0
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)

def Transform(T, p):
    return np.dot(T[:3, :3], p.T).T + T[:3, 3]

def CalcVolume(fname):
    print("1. loading original ply file...")
    pc = o3d.io.read_point_cloud(fname)
    pts = np.array(pc.points)
    print("2. loading manually selected 4 points...")
    gpts = np.loadtxt("../Data/picking_list.txt", delimiter=",") # load selected ground points
    print("3. constucting an space from 4 selected points and transform all points into the space...")
    center = gpts[0]
    mean = gpts.mean(axis = 0)
    zv = np.linalg.svd(gpts - mean)[2][-1]
    xv = gpts[0] - gpts[1]
    xv /= np.linalg.norm(xv)
    yv = np.cross(zv, xv)
    xv = np.cross(zv, yv)
    T = np.eye(4)
    T[:3, 0] = xv
    T[:3, 1] = yv
    T[:3, 2] = zv
    T[:3, 3] = center
    iT = np.linalg.inv(T)
    npts = Transform(iT, pts)
    ngpts = Transform(iT, gpts)

    print("4. reconstructing surface by using poisson...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         vertex_normals=np.asarray(mesh.vertex_normals))
    
    print("5. calculate the volume by summarizing all triangle-plane volumes...")
    pp4 = Polygon(ngpts[:, :2]).convex_hull
    
    volume = 0.0        
    for t in tm.triangles:
        nt = Transform(iT, t)
        mp = MultiPoint(nt[:, :2].tolist())        
        if pp4.contains(mp):            
            volume += TriVolume(nt)
    print("The volume of the mound is: {0} m^3".format(volume))
    
if __name__ == '__main__':
    CalcVolume("../Data/0.ply")
