package gr.uth.ece.dsel.spark.util;

import java.util.Iterator;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

public final class CellContainsQueryPoints implements Function<Tuple2<String, Tuple2<Iterable<Point>, Iterable<Point>>>, Boolean>
{
	@Override
	public final Boolean call(Tuple2<String, Tuple2<Iterable<Point>, Iterable<Point>>> qtIterable)
	{
		Iterator<Point> qpoint = qtIterable._2._1.iterator(); // iterator on query points tuples
		
		if (qpoint.hasNext())
			return true;
		else
			return false;
	}
}
