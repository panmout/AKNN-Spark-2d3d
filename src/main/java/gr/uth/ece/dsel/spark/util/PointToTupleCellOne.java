package gr.uth.ece.dsel.spark.util;

import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

public final class PointToTupleCellOne implements PairFunction<Point, String, Integer>
{
	private int N = 0;
	private Node node = null;
	
	public PointToTupleCellOne (int n)
	{
		this.N = n;
	}
	
	public PointToTupleCellOne (Node node)
	{
		this.node = node;
	}
	
	@Override
	public final Tuple2<String, Integer> call(Point p)
	{
		String cell = null;
		if (this.N != 0)
			cell = AknnFunctions.pointToCellGD(p, this.N);
		else if (this.node != null)
		{
			if (this.node.getCNE() == null) // 2d
				cell = AknnFunctions.pointToCellQT(p.getX(), p.getY(), this.node);
			else // 3d
				cell = AknnFunctions.pointToCellQT(p.getX(), p.getY(), p.getZ(), this.node);
		}
		return new Tuple2<String, Integer>(cell, 1);
	}
}
