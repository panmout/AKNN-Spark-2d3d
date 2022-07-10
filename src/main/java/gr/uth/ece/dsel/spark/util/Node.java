package gr.uth.ece.dsel.spark.util;

import java.io.Serializable;
import java.util.HashSet;

public final class Node implements Serializable
{
	//private int low, high; // lower, higher index of sample array
	private Node nw, ne, sw, se; // 2d children
	private Node cnw, cne, csw, cse, fnw, fne, fsw, fse; // 3d children (c: ceiling, f: floor)
	private double xmin, xmax, ymin, ymax, zmin, zmax; // node boundaries
	private HashSet<Integer> contPoints = new HashSet<Integer>(); // points contained
	
	// constructor
	public Node (double xmin, double ymin, double xmax, double ymax)
	{
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
	}
	
	// constructor
	public Node (double xmin, double ymin, double xmax, double ymax, double zmin, double zmax)
	{
		// The node will be defined using the coordinates of two opposite corners
		// Floor South West (xmin, ymin, zmin) and
		// Ceiling North East (xmax, ymax, zmax)
		this.xmin = xmin;
		this.xmax = xmax;
		this.ymin = ymin;
		this.ymax = ymax;
		this.zmin = zmin;
		this.zmax = zmax;
	}
	/*
	public final int getLow()
	{
		return this.low;
	}

	public final void setLow(int low)
	{
		this.low = low;
	}

	public final int getHigh()
	{
		return this.high;
	}

	public final void setHigh(int high)
	{
		this.high = high;
	}
	*/
	
	// 2d set-get
	public final double getXmin()
	{
		return this.xmin;
	}

	public final void setXmin(double xmin)
	{
		this.xmin = xmin;
	}

	public final double getXmax()
	{
		return this.xmax;
	}

	public final void setXmax(double xmax)
	{
		this.xmax = xmax;
	}

	public final double getYmin()
	{
		return this.ymin;
	}

	public final void setYmin(double ymin)
	{
		this.ymin = ymin;
	}

	public final double getYmax()
	{
		return this.ymax;
	}

	public final void setYmax(double ymax)
	{
		this.ymax = ymax;
	}
	
	public final Node getNW()
	{
		return this.nw;
	}

	public final void setNW(Node nW)
	{
		this.nw = nW;
	}

	public final Node getNE()
	{
		return this.ne;
	}

	public final void setNE(Node nE)
	{
		this.ne = nE;
	}

	public final Node getSW()
	{
		return this.sw;
	}

	public final void setSW(Node sW)
	{
		this.sw = sW;
	}

	public final Node getSE()
	{
		return this.se;
	}

	public final void setSE(Node sE)
	{
		this.se = sE;
	}
	
	// 3d set-get
	public final double getZmin()
	{
		return this.zmin;
	}

	public final void setZmin(double zmin)
	{
		this.zmin = zmin;
	}

	public final double getZmax()
	{
		return this.zmax;
	}

	public final void setZmax(double zmax)
	{
		this.zmax = zmax;
	}
	
	public final Node getFNW()
	{
		return this.fnw;
	}

	public final void setFNW(Node fnw)
	{
		this.fnw = fnw;
	}

	public final Node getFNE()
	{
		return this.fne;
	}

	public final void setFNE(Node fne)
	{
		this.fne = fne;
	}

	public final Node getFSW()
	{
		return this.fsw;
	}

	public final void setFSW(Node fsw)
	{
		this.fsw = fsw;
	}

	public final Node getFSE()
	{
		return this.fse;
	}

	public final void setFSE(Node fse)
	{
		this.fse = fse;
	}
	
	public final Node getCNW()
	{
		return this.cnw;
	}

	public final void setCNW(Node cnw)
	{
		this.cnw = cnw;
	}

	public final Node getCNE()
	{
		return this.cne;
	}

	public final void setCNE(Node cne)
	{
		this.cne = cne;
	}

	public final Node getCSW()
	{
		return this.csw;
	}

	public final void setCSW(Node csw)
	{
		this.csw = csw;
	}

	public final Node getCSE()
	{
		return this.cse;
	}

	public final void setCSE(Node cse)
	{
		this.cse = cse;
	}
	
	// contained point methods
	public final void addPoints(int i)
	{
		this.contPoints.add(i);
	}
	
	public final void removePoints()
	{
		// 2d
		if (this.nw != null)
		{
			this.nw.removePoints();
			this.nw.contPoints.clear();
		}
		if (this.ne != null)
		{
			this.ne.removePoints();
			this.ne.contPoints.clear();
		}
		if (this.sw != null)
		{
			this.sw.removePoints();
			this.sw.contPoints.clear();
		}
		if (this.se != null)
		{
			this.se.removePoints();
			this.se.contPoints.clear();
		}
		
		// 3d
		if (this.cnw != null)
		{
			this.cnw.removePoints();
			this.cnw.contPoints.clear();
		}
		if (this.cne != null)
		{
			this.cne.removePoints();
			this.cne.contPoints.clear();
		}
		if (this.csw != null)
		{
			this.csw.removePoints();
			this.csw.contPoints.clear();
		}
		if (this.cse != null)
		{
			this.cse.removePoints();
			this.cse.contPoints.clear();
		}
		if (this.fnw != null)
		{
			this.fnw.removePoints();
			this.fnw.contPoints.clear();
		}
		if (this.fne != null)
		{
			this.fne.removePoints();
			this.fne.contPoints.clear();
		}
		if (this.fsw != null)
		{
			this.fsw.removePoints();
			this.fsw.contPoints.clear();
		}
		if (this.fse != null)
		{
			this.fse.removePoints();
			this.fse.contPoints.clear();
		}
	}

	public final HashSet<Integer> getContPoints()
	{
		return this.contPoints;
	}

	public final void setContPoints(HashSet<Integer> contPoints)
	{
		this.contPoints = new HashSet<Integer>(contPoints);
	}
	
	public final void removePoint(int i)
	{
		this.contPoints.remove(i);
	}
}
