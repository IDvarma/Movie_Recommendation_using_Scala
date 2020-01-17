// Databricks notebook source
/* Data is already loaded into Data Bricks else import can be done as below

Data Import:
------------
val path = "file:///Downloads/movie.csv"
val movie = spark.read.format("csv").option("header", "true").load(path)
val movie1 = movie.selectExpr("cast(movieId as int) movieId", "cast(title as string) title", "cast(genres as string) genres")

*/

// COMMAND ----------

val movie = spark.sql("select * from movie1")
movie.show(10)

// COMMAND ----------

/* Data is already loaded into Data Bricks else import can be done as below

Data Import:
------------
val path = "file:///Downloads/rating.csv"
val rating = spark.read.format("csv").option("header", "true").load(path)
val rating1 = rating.selectExpr("cast(userId as int) userId","cast(movieId as int) movieId" ,"cast(rating as int) rating", "cast(timestamp as date) timestamp")

*/

// COMMAND ----------

val rating = spark.sql("select * from rating1")
rating.show(10)

// COMMAND ----------

/*Register both DataFrames as temp tables to make querying easier */
rating.createOrReplaceTempView("ratings")
movie.createOrReplaceTempView("movies")

// COMMAND ----------

/* Count of ratings, users, movies */
val numRatings = rating.count()
val numUsers = rating.select(rating.col("userId")).distinct().count()
val numMovies = rating.select(rating.col("movieId")).distinct().count() 
println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.") 

// COMMAND ----------

/* Get the max, min ratings along with the count of users who have rated a movie */
val results = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "
+ "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"
+ "min(ratings.rating) as minr,count(distinct userId) as cntu "
+ "FROM ratings group by ratings.movieId) movierates "
+ "join movies on movierates.movieId=movies.movieId "
+ "order by movierates.cntu desc") 
results.show()

// COMMAND ----------

/* 10 most active users and how many times they rated a movie */
val mostActiveUsers = spark.sql("select userid, count(userid) as count from ratings group by userid order by count desc limit 10")
mostActiveUsers.show()

// COMMAND ----------

/* particular user and find the movies that, say user, 118205 rated higher than 4 */

val results2 = spark.sql(" select ratings.userid,ratings.movieid,ratings.rating,movies.title from ratings join movies on ratings.movieid = movies.movieid where ratings.rating > 4 and ratings.userid = 118205")
results2.show()

// COMMAND ----------

/* Importing Libraries */
import spark.implicits._

import org.apache.spark.ml.recommendation.ALS

/* Build ALS() model */

val als = new ALS().
  setMaxIter(5).
  setRegParam(0.01).
  setUserCol("userId").
  setItemCol("movieId").
  setRatingCol("rating")
println(als.explainParams())

// COMMAND ----------

/* Split data into training and test */
import org.apache.spark.ml.recommendation.ALS.Rating
val Array(training, testing) = rating.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

/* Fitting Model */
import org.apache.spark.ml.recommendation.ALSModel
val model = als.fit(training)

// COMMAND ----------

/* Prediction */
/* Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics */
model.setColdStartStrategy("drop")
val predictions = model.transform(testing)
predictions.show(10)

// COMMAND ----------

/* Model Evaluation */

import org.apache.spark.ml.evaluation.RegressionEvaluator
val evaluator = new RegressionEvaluator().
  setMetricName("rmse").  // root mean squared error
  setLabelCol("rating").
  setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

// COMMAND ----------

import org.apache.spark.mllib.evaluation.{
  RankingMetrics,
  RegressionMetrics}
val regComparison = predictions.select("rating", "prediction")
  .rdd.map(x => (x.getFloat(0).toDouble,x.getFloat(1).toDouble))
val metrics = new RegressionMetrics(regComparison)

// COMMAND ----------

/* Any movie a user rated higher than 2.5 */
import org.apache.spark.sql.functions.{col, expr}
val perUserActual = predictions
  .where("rating > 2.5")
  .groupBy("userId")
  .agg(expr("collect_set(movieId) as movies"))

// COMMAND ----------

val perUserPredictions = predictions
  .orderBy(col("userId"), col("prediction").desc)
  .groupBy("userId")
  .agg(expr("collect_list(movieId) as movies"))

// COMMAND ----------

val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("userId"))
  .map(row => (
    row(1).asInstanceOf[Seq[Integer]].toArray,
    row(2).asInstanceOf[Seq[Integer]].toArray.take(15)
  ))
val ranks = new RankingMetrics(perUserActualvPred.rdd)

// COMMAND ----------

ranks.meanAveragePrecision
ranks.precisionAt(5)

// COMMAND ----------



// COMMAND ----------

// Model is ready for recommendations

// Generate top 10 movie recommendations for each user
val userRecs = model.recommendForAllUsers(10)
userRecs.show(truncate = false)

// COMMAND ----------

// Generate top 10 user recommendations for each movie
val movieRecs = model.recommendForAllItems(10)
movieRecs.show(truncate = false)

// COMMAND ----------

// Generate top 10 movie recommendations for a specified set of users
// Use a trick to make sure we work with the known users from the input
val users = rating.select(als.getUserCol).distinct.limit(20)
val userSubsetRecs = model.recommendForUserSubset(users, 10)
userSubsetRecs.show(truncate = false)

// COMMAND ----------

// Generate top 10 user recommendations for a specified set of movies
val movies = rating.select(als.getItemCol).distinct.limit(3)
val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movieSubSetRecs.show(truncate = false)
