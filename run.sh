partitioning=gd # gd or qt
method=ps # bf or ps
K=10
N=500
NameNode=Hadoopmaster
queryDir=input
trainingDir=input
queryDataset=all_buildingsNNew_obj_5.txt
trainingDataset=paskrsNNew_obj.txt
treeDir=sampletree
treeFileName=qtree.ser
partitions=128

spark-submit \
--class gr.uth.ece.dsel.spark.main.Main \
./target/aknn-spark-2d3d-0.0.1-SNAPSHOT.jar \
$partitioning $method $K $N $NameNode $queryDir $queryDataset $trainingDir $trainingDataset $treeDir $treeFileName $partitions
