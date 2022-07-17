package gr.uth.ece.dsel.spark.main;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.FormatterClosedException;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;

import scala.Tuple2;
import gr.uth.ece.dsel.spark.util.*;

public class Main
{
	private static Formatter outputTextFile; // local output text file
	
	public static void main(String[] args)
	{
		final long t0 = System.currentTimeMillis();
		/*
	     **************************************************************************
	     *                                                                        *
	     *                         Class variables,                               *
	     *                      command line arguments,                           *
	     *                         open HDFS files                                *
	     *                                                                        *
	     **************************************************************************                                 
	     */
		
		try
		{
			outputTextFile = new Formatter(new FileWriter("results.txt", true)); // appendable file
		}
		catch (IOException ioException)
		{
			System.err.println("Could not open file, exiting");
			System.exit(1);
		}
		
		// args
		final String partitioning = args[0];
		final String method = args[1];
		final int K = Integer.parseInt(args[2]);
		final int N = Integer.parseInt(args[3]);
		final String nameNode = args[4];
		final String queryDir = args[5];
		final String queryDataset = args[6];
		final String trainingDir = args[7];
		final String trainingDataset = args[8];
		final String treeDir = args[9]; // HDFS dir containing tree file
		final String treeFileName = args[10]; // tree file name in HDFS
		final int partitions = Integer.parseInt(args[11]); // set partitions to total number of cores
		
		final String username = System.getProperty("user.name"); // get user name
		
		if (!partitioning.equals("qt") && !partitioning.equals("gd"))
			throw new IllegalArgumentException("partitoning arg must be 'qt' or 'gd'");
		
		if (!method.equals("bf") && !method.equals("ps"))
			throw new IllegalArgumentException("method arg must be 'bf' or 'ps'");
		
		String startMessage = String.format("AKNN %s-%s starting...\n", partitioning.toUpperCase(), method.toUpperCase());
		System.out.println(startMessage);
		writeToFile(outputTextFile, startMessage);
		
		// create HDFS files paths
	    final String queryFile = String.format("hdfs://%s:9000/user/%s/%s/%s", nameNode, username, queryDir, queryDataset);
	    final String trainingFile = String.format("hdfs://%s:9000/user/%s/%s/%s", nameNode, username, trainingDir, trainingDataset);
	    final String treeFile = String.format("hdfs://%s:9000/user/%s/%s/%s", nameNode, username, treeDir, treeFileName); // full HDFS path to tree file
	    
	    // display arguments to the console
	    String arguments = String.format("partitioning=%s\nmethod=%s\nK=%d\nQuery Dataset=%s\nTraining Dataset=%s\n", partitioning, method, K, queryFile, trainingFile);
	  	if (partitioning.equals("gd"))
	  		arguments += "N=" + N + "\n";
	  	else if (partitioning.equals("qt"))
	  		arguments += "sampletree=" + treeFileName + "\n";
	    System.out.println("Input arguments: \n" + arguments);
	    
	    // Spark conf
	  	SparkConf sparkConf = new SparkConf().setAppName("aknn-spark-2d3d").setMaster(String.format("spark://%s:7077", nameNode)); //("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		
		// Hadoop FS
		Configuration hadoopConf = new Configuration();
	    hadoopConf.set("fs.defaultFS", String.format("hdfs://%s:9000", nameNode));
	    FileSystem fs;
	    
	    Node root = null;
		
	    // open HDFS files
	  	try
	  	{
	  		fs = FileSystem.get(hadoopConf);
	  		
	  		// read treefile from hdfs
			if (partitioning.equals("qt"))
			{
				Path pt = new Path(treeFile); // create path object from path string
				ObjectInputStream input = new ObjectInputStream(fs.open(pt)); // open HDFS tree file
				root = (Node) input.readObject(); // assign quad tree binary form to root node
			}
	  	}
	  	catch (ClassNotFoundException classNotFoundException)
		{
			System.err.println("Invalid object type");
		}
	  	catch (IOException e)
		{
			System.err.println("hdfs file does not exist");
			e.printStackTrace();
		}
	  	
	  	//create query and training point RDDs
	  	JavaRDD<Point> qpointsRDD = jsc.textFile(queryFile, partitions)
	  								   .map(line -> AknnFunctions.newPoint(line, "\t")) // map a line <id, x, y> to a point object with neighbors list and boolean status
	  								   .persist(StorageLevel.MEMORY_AND_DISK());
	  	
	  	JavaRDD<Point> tpointsRDD = jsc.textFile(trainingFile, partitions)
	  								   .map(line -> AknnFunctions.newPoint(line, "\t")) // map a line <id, x, y> to a point object
	  								   .persist(StorageLevel.MEMORY_AND_DISK());
	  	
	  	/*
	     **************************************************************************
	     *                             PHASE 1                                    *
	     *                                                                        *
	     *             find number of training points per cell                    *
	     **************************************************************************                                 
	     */
	    
	    final long t1 = System.currentTimeMillis();
	    System.out.println("PHASE 1 starting...");
	    
	    // get map(cell_id, contained training points)
	    JavaPairRDD<String, Integer> tPointsPerCellRDD = JavaPairRDD.fromJavaRDD(jsc.emptyRDD());
	    
	    if (partitioning.equals("gd"))
	    	tPointsPerCellRDD = tpointsRDD.mapToPair(new PointToTupleCellOne(N))    // map each point to a tuple <cell, 1>
		    							  .reduceByKey((a, b) -> a + b, partitions); // for each cell, add the 1's
		    							  //.persist(StorageLevel.MEMORY_AND_DISK());
	    else if (partitioning.equals("qt"))
	    	tPointsPerCellRDD = tpointsRDD.mapToPair(new PointToTupleCellOne(root)) // map each point to a tuple <cell, 1>
										  .reduceByKey((a, b) -> a + b, partitions) // for each cell, add the 1's
										  .persist(StorageLevel.MEMORY_AND_DISK());
	    
	    System.out.printf("number of cells = %d\n", tPointsPerCellRDD.count());
	    
	    System.out.printf("tPointsPerCellRDD numPartitions: %d%n", tPointsPerCellRDD.getNumPartitions());
	    
	    // collect Phase 1 output: HashMap(cell_id, num tpoints)
	    final HashMap<String, Integer> cell_tpoints = new HashMap<String, Integer>(tPointsPerCellRDD.collectAsMap());
	    
	    String phase1Message = String.format("PHASE 1 finished in %d millis\n", System.currentTimeMillis() - t1);
	    System.out.println(phase1Message);
		writeToFile(outputTextFile, phase1Message);
	    
	    /*
	     **************************************************************************
	     *                             PHASE 2                                    *
	     *                                                                        *
	     *              discover neighbors inside each cell                       *
	     **************************************************************************
	     */
	    
	    final long t2 = System.currentTimeMillis();
	    System.out.println("PHASE 2 starting...");
	    
	    // create pair RDDs [cell, point]
	    JavaPairRDD<String, Point> cellQpointsRDD = JavaPairRDD.fromJavaRDD(jsc.emptyRDD());
	    
	    JavaPairRDD<String, Point> cellTpointsRDD = JavaPairRDD.fromJavaRDD(jsc.emptyRDD());
	    
	    if (partitioning.equals("gd"))
	    {
	    	cellQpointsRDD = qpointsRDD.mapToPair(new PointToTupleCellPoint(N)); // map each point to a tuple <cell, point>
	    	
	    	cellTpointsRDD = tpointsRDD.mapToPair(new PointToTupleCellPoint(N)) // map each point to a tuple <cell, point>
					   				   .persist(StorageLevel.MEMORY_AND_DISK());
	    }
	    else if (partitioning.equals("qt"))
	    {
	    	cellQpointsRDD = qpointsRDD.mapToPair(new PointToTupleCellPoint(root)); // map each point to a tuple <cell, point>

	    	cellTpointsRDD = tpointsRDD.mapToPair(new PointToTupleCellPoint(root)) // map each point to a tuple <cell, point>
									   .persist(StorageLevel.MEMORY_AND_DISK());
	    }
	    
	    // join Q and T datasets by cell
	    JavaPairRDD<String, Tuple2<Iterable<Point>, Iterable<Point>>> cellQTjoinRDD = cellQpointsRDD.cogroup(cellTpointsRDD);
	    
	    // find neighbors
	    
	    // RDD of cells, query points and their neighbor lists for phase 2
	    JavaPairRDD<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>> neighbors2RDD = JavaPairRDD.fromJavaRDD(jsc.emptyRDD());
	    
	    if (method.equals("bf"))
	    	neighbors2RDD = cellQTjoinRDD.filter(new CellContainsQueryPoints()) // only cells with query points pass
	    								 .mapValues(new BfNeighbors(K))         // find neighbors
	    								 .persist(StorageLevel.MEMORY_AND_DISK());
	    else if (method.equals("ps"))
	    	neighbors2RDD = cellQTjoinRDD.filter(new CellContainsQueryPoints()) // only cells with query points pass
	    								 .mapValues(new PsNeighbors(K))         // find neighbors
	    								 .persist(StorageLevel.MEMORY_AND_DISK());
	    
//	    StringBuilder sb = new StringBuilder();
//	    
//	    List<Tuple2<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>>> neighbors2 = neighbors2RDD.collect();
//	    
//	    for (Tuple2<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>> e : neighbors2)
//	    {
//	    	for (Tuple2<Point, PriorityQueue<IdDist>> tuple : e._2)
//	    	{
//	    		sb.append(String.format("%d\t%s\t%s\n", tuple._1.getId(), e._1, AknnFunctions.pqToString(tuple._2, K, "min")));
//	    	}
//	    }
//	    
//	    WriteLocalFiles.writeFile("neighbors2.txt", sb.toString());
	    
	    System.out.printf("neighbors2RDD size: %d\n", neighbors2RDD.count());
	    
	    String phase2Message = String.format("PHASE 2 finished in %d millis\n", System.currentTimeMillis() - t2);
	    System.out.println(phase2Message);
		writeToFile(outputTextFile, phase2Message);
		
	    /*
	     **************************************************************************
	     *                             PHASE 3                                    *
	     *                                                                        *
	     *              discover neighbors in neighboring cells                   *
	     **************************************************************************
	    */
	    
	    final long t3 = System.currentTimeMillis();
	    System.out.println("PHASE 3 starting...");
	    
	    GetOverlaps getOverlaps = new GetOverlaps(cell_tpoints, K, partitioning);
	    
	    if (partitioning.equals("gd"))
	    	getOverlaps.setN(N);
	    else if (partitioning.equals("qt"))
	    	getOverlaps.setRoot(root);
	    
	    // create RDD <overlapped cell, list of <qpoint, true/false>>
	    JavaPairRDD<String, Tuple2<Point, Boolean>> overlapsRDD = neighbors2RDD.map(getOverlaps) 		   // output list with tuples <cell, <qpoint, true/false>
						    												   .flatMap(e -> e.iterator()) // flatten lists into their elements
						    												   .mapToPair(e -> new Tuple2<String, Tuple2<Point, Boolean>>(e._1, new Tuple2<Point, Boolean>(e._2._1, e._2._2)));
	    
//	    // RDD of "true" query points (with complete neighbor list)
//	    JavaPairRDD<String, Point> qpointsRDDtrue = overlapsRDD.filter(e -> e._2._2 == true) // only "true" qpoints pass
//																.mapValues(e -> e._1); 		 // keep qpoint, drop boolean
	    
	    // RDD of "false" query points (with incomplete neighbor list)
	    JavaPairRDD<String, Point> qpointsRDDfalse = overlapsRDD.filter(e -> e._2._2 == false) // only "false" qpoints pass
																.mapValues(e -> e._1); 		   // keep qpoint, drop boolean
	    
	    // RDD of cells, query points and their neighbor lists for phase 2
	    JavaPairRDD<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>> neighbors3RDD = JavaPairRDD.fromJavaRDD(jsc.emptyRDD());
	    
	    if (method.equals("bf"))
	    	neighbors3RDD = qpointsRDDfalse.cogroup(cellTpointsRDD)			  // join tpoints by cell RDD
	    							   	   .mapValues(new BfNeighbors(K))     // find neighbors
	    							   	   .persist(StorageLevel.MEMORY_AND_DISK());
	    else if (method.equals("ps"))
	    	neighbors3RDD = qpointsRDDfalse.cogroup(cellTpointsRDD)			  // join tpoints by cell RDD
	    							       .mapValues(new PsNeighbors(K))     // find neighbors
	    							       .persist(StorageLevel.MEMORY_AND_DISK());
	    
//	    StringBuilder sb = new StringBuilder();
//	    
//	    List<Tuple2<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>>> neighbors3 = neighbors3RDD.collect();
//	    
//	    for (Tuple2<String, ArrayList<Tuple2<Point, PriorityQueue<IdDist>>>> tuple: neighbors3)
//	    	for (Tuple2<Point, PriorityQueue<IdDist>> e : tuple._2)
//	    		sb.append(String.format("%s\t%d\t%s\n", tuple._1, e._1.getId(), AknnFunctions.pqToString(e._2, K, "min")));
//	    
//	    WriteLocalFiles.writeFile("neighbors3.txt", sb.toString());
	    
	    System.out.printf("neighbors3RDD size: %d\n", neighbors3RDD.count());
	    
	    String phase3Message = String.format("PHASE 3 finished in %d millis\n", System.currentTimeMillis() - t3);
	    System.out.println(phase3Message);
		writeToFile(outputTextFile, phase3Message);
		
	    /*
	     **************************************************************************
	     *                             PHASE 4                                    *
	     *                                                                        *
	     *              merge neighbor lists from phases 2 and 3                  *
	     **************************************************************************
	     */
	    
	    final long t4 = System.currentTimeMillis();
	    System.out.println("PHASE 4 starting...");
	    
	    // RDD of query points and their neighbor lists from phase 2
	    JavaPairRDD<Integer, PriorityQueue<IdDist>> neighbors2RDDnew = neighbors2RDD.values()	// drop cells
	    																		  	.flatMap(e -> e.iterator()) // flatten list to its elements (tuples of qpoints and neighbor lists)
	    																		  	.mapToPair(e -> new Tuple2<Integer, PriorityQueue<IdDist>>(e._1.getId(), e._2)); // create pair RDD <qpoint id, priority queue>
	    
//	    List<Tuple2<Integer, PriorityQueue<IdDist>>> neighbors2 = neighbors2RDDnew.take(10);
//	    System.out.println("neighbors2");
//	    for (Tuple2<Integer, PriorityQueue<IdDist>> tuple: neighbors2)
//	    	System.out.printf("qpoint: %d\tneighbors: %s\n", tuple._1, AknnFunctions.pqToString(tuple._2, K, "min"));
	    
	    // RDD of query points and their merged multiple neighbor lists from phase 3 ("false" case)
	    JavaPairRDD<Integer, PriorityQueue<IdDist>> neighbors3RDDnew = neighbors3RDD.values()	// drop cells
				  																  	.flatMap(e -> e.iterator()) // flatten list to its elements (tuples of qpoints and neighbor lists)
				  																  	.mapToPair(e -> new Tuple2<Integer, PriorityQueue<IdDist>>(e._1.getId(), e._2)) // create pair RDD <qpoint id, priority queue>
				  																  	.reduceByKey((pq1, pq2) -> AknnFunctions.joinPQ(pq1, pq2, K)); // merge multiple priority queues by qpoint id
	    
//	    List<Tuple2<Integer, PriorityQueue<IdDist>>> neighbors3 = neighbors3RDDnew.take(10);
//	    System.out.println("neighbors3");
//	    for (Tuple2<Integer, PriorityQueue<IdDist>> tuple: neighbors3)
//	    	System.out.printf("qpoint: %d\t%s\n", tuple._1, AknnFunctions.pqToString(tuple._2, K, "min"));
	    
	    // create RDD with all neighbor lists per qpoint from phases 2 & 3
	    JavaPairRDD<Integer, Tuple2<Iterable<PriorityQueue<IdDist>>, Iterable<PriorityQueue<IdDist>>>> neighbors23RDD = neighbors2RDDnew.cogroup(neighbors3RDDnew);
	    
	    JavaPairRDD<Integer, PriorityQueue<IdDist>> finalNeighborsRDD = neighbors23RDD.mapValues(new MergePQ(K));
	    
	    
	    StringBuilder sb = new StringBuilder();
	    
	    List<Tuple2<Integer, PriorityQueue<IdDist>>> finalNeighbors = finalNeighborsRDD.take(1000);
	    
	    for (Tuple2<Integer, PriorityQueue<IdDist>> tuple : finalNeighbors)
	    	sb.append(String.format("%d\t%s\n", tuple._1, AknnFunctions.pqToString(tuple._2, K, "min")));
	    
	    WriteLocalFiles.writeFile("final_neighbors.txt", sb.toString());
	    
//	    // create RDD with best neighbors per query point
//	    JavaPairRDD<Point, PriorityQueue<IdDist>> finalNeighbors = neighbors23RDD.mapValues(tuple -> AknnFunctions.joinPQ(tuple._1, tuple._2, K));
	    
	    String phase4Message = String.format("PHASE 4 finished in %d millis\n", System.currentTimeMillis() - t4);
	    System.out.println(phase4Message);
		writeToFile(outputTextFile, phase4Message);
		
		String finalMessage = String.format("%s-%s finished in %d millis\n", partitioning.toUpperCase(), method.toUpperCase(), System.currentTimeMillis() - t0);
		System.out.println(finalMessage);
		writeToFile(outputTextFile, finalMessage);
		
	    jsc.close();
	}
	
	private static void writeToFile(Formatter file, String s)
	{
		try
		{
			outputTextFile.format(s);
		}
		catch (FormatterClosedException formatterException)
		{
			System.err.println("Error writing to file, exiting");
			System.exit(1);
		}
	}
}
