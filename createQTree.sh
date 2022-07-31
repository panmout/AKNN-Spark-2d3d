###########################################################################
#                             PARAMETERS                                  #
###########################################################################

nameNode=Hadoopmaster
trainingDir=input
treeDir=sampletree
trainingDataset=paskrsNNew_obj_3d.txt
samplerate=1
capacity=200
type=1 # 1 for simple capacity based quadtree, 2 for all children split method, 3 for average width method

###########################################################################
#                                    EXECUTE                              # ###########################################################################

spark-submit \
--class gr.uth.ece.dsel.spark.util.Qtree \
./target/aknn-spark-2d3d-0.0.1-SNAPSHOT.jar \
nameNode=$nameNode \
trainingDir=$trainingDir \
treeDir=$treeDir \
trainingDataset=$trainingDataset \
samplerate=$samplerate \
capacity=$capacity \
type=$type
