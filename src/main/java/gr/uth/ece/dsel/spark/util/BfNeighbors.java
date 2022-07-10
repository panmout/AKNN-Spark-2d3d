package gr.uth.ece.dsel.spark.util;

import java.util.ArrayList;
import java.util.PriorityQueue;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

public final class BfNeighbors implements Function<Tuple2<Iterable<Point>, Iterable<Point>>, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>>
{
	private int k;
	private PriorityQueue<IdDist> neighbors;
	private ArrayList<Tuple2<Point, PriorityQueue<IdDist>>> qpoint_neighbors;
	
	public BfNeighbors(int k)
	{
		this.k = k;
		this.neighbors = new PriorityQueue<IdDist>(this.k, new IdDistComparator("max"));
		this.qpoint_neighbors = new ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>();
	}
	
	@Override
	public final ArrayList<Tuple2<Point, PriorityQueue<IdDist>>> call (Tuple2<Iterable<Point>, Iterable<Point>> tuple)
	{
		this.qpoint_neighbors.clear();
		
		// traverse <query points, neighbors> tuples
	    for (Point qpoint: tuple._1)
	    {
    		// traverse training points
	    	for (Point tpoint: tuple._2)
	    	{
	    		final double dist = AknnFunctions.distance(qpoint, tpoint); // distance calculation
		    	final IdDist neighbor = new IdDist(tpoint.getId(), dist); // create neighbor
		    	
		    	// if PriorityQueue not full, add new tpoint (IdDist)
		    	if (this.neighbors.size() < this.k)
		    	{
			    	if (!AknnFunctions.isDuplicate(this.neighbors, neighbor))
			    		this.neighbors.offer(neighbor); // insert to queue
		    	}
		    	else // if queue is full, run some checks and replace elements
		    	{
		    		final double dm = this.neighbors.peek().getDist(); // get (not remove) distance of neighbor with maximum distance
		    		
	  				if (dist < dm) // compare distance
	  				{  					
	  					if (!AknnFunctions.isDuplicate(this.neighbors, neighbor))
	  					{
	  						this.neighbors.poll(); // remove top element
	  			    		this.neighbors.offer(neighbor); // insert to queue
	  					}
	  				} // end if
		    	} // end else
			} // end training points traverse
	    	//System.out.printf("qpoint: %d, neighbors: %s\n", qpoint.getId(), AknnFunctions.pqToString(this.neighbors, this.k, "min"));
	    	this.qpoint_neighbors.add(new Tuple2<Point, PriorityQueue<IdDist>>(qpoint, new PriorityQueue<IdDist>(this.neighbors)));
	    	this.neighbors.clear();
	    } // end query points traverse
	    // return <list of qpoints with their neighbor list>
	    return new ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>(this.qpoint_neighbors);
	} // end gdBfNeighbors
}
