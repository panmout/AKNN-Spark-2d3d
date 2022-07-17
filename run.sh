partitioning=gd # gd or qt
method=ps # bf or ps
K=10
N=10
NameNode=panagiotis-lubuntu
queryDir=input
trainingDir=input
queryDataset=query-dataset3d.txt
trainingDataset=NApppointNNew3d.txt
treeDir=sampletree
treeFileName=qtree.ser
partitions=2

spark-submit \
--class gr.uth.ece.dsel.spark.main.Main \
./target/aknn-spark-2d3d-0.0.1-SNAPSHOT.jar \
$partitioning $method $K $N $NameNode $queryDir $queryDataset $trainingDir $trainingDataset $treeDir $treeFileName $partitions
