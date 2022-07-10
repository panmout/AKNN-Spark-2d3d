package gr.uth.ece.dsel.spark.util;

import java.util.PriorityQueue;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

public class MergePQ implements Function<Tuple2<Iterable<PriorityQueue<IdDist>>, Iterable<PriorityQueue<IdDist>>>, PriorityQueue<IdDist>>
{
	private int k;
	private PriorityQueue<IdDist> neighbors;
	
	public MergePQ (int k)
	{
		this.k = k;
		this.neighbors = new PriorityQueue<IdDist>(this.k, new IdDistComparator("max"));
	}
	
	@Override
	public PriorityQueue<IdDist> call (Tuple2<Iterable<PriorityQueue<IdDist>>, Iterable<PriorityQueue<IdDist>>> tuple)
	{
		this.neighbors.clear();
		
		// first iterable of priority queues are neighbor lists from phase 2
		for (PriorityQueue<IdDist> pq1 : tuple._1)
			joinPQ(pq1);
			
		// second iterable of priority queues are neighbor lists from phase 3
		for (PriorityQueue<IdDist> pq2 : tuple._2)
			joinPQ(pq2);
		
		while (this.neighbors.size() > this.k)
			this.neighbors.poll();
		
		return new PriorityQueue<IdDist>(this.neighbors);
	}
	
	private void joinPQ (PriorityQueue<IdDist> pq)
	{
		while (!pq.isEmpty())
		{
			IdDist neighbor = pq.poll();
			if (!AknnFunctions.isDuplicate(this.neighbors, neighbor))
				this.neighbors.add(neighbor);
		}
	}
}
