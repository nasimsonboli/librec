The added classes are BLNSLIMUserRecommender and SLIMUser Recommender 
in this path "librec/core/src/main/java.net.librec/recommender/cf/ranking/"

A guide named "blnslim-test.properties" for the properties file is in this path:
"librec/core/src/main/resources/rec/cf/ranking/"

We need to add a membership file to the data path as well. There is a membership folder
 in the data path. Any membership file should be added there. A membership file format
for each row is "[userid]\t[membership]\r\n", while membership is either -1 or 1.
