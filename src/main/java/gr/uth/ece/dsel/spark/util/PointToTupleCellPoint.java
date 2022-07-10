package gr.uth.ece.dsel.spark.util;

import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

public final class PointToTupleCellPoint implements PairFunction<Point, String, Point>
{
	private int N = 0;
	private Node node = null;
	
	public PointToTupleCellPoint (int n)
	{
		this.N = n;
	}
	
	public PointToTupleCellPoint (Node node)
	{
		this.node = node;
	}
	
	@Override
	public final Tuple2<String, Point> call(Point p)
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
		return new Tuple2<String, Point>(cell, p);
	}
}
