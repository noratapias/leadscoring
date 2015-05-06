package org.template.leadscoring

import io.prediction.controller.PPreparator
//import io.prediction.data.storage.BiMap

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import grizzled.slf4j.Logger

class PreparedData(
  val labeledPoints: RDD[LabeledPoint],
  val featureIndex: Map[String, Int],
  val featureCategoricalIntMap: Map[String, Map[String, Int]]
) extends Serializable

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  private def createCategoricalIntMap(
    values: Array[String], // categorical values
    default: String // default cateegorical value
  ): Map[String, Int] = {
    val m = values.zipWithIndex.toMap
    if (m.contains(default))
      m
    else
      // add default value if origina values don't have it
      m + (default -> m.size)
  }

  def prepare(sc: SparkContext, td: TrainingData): PreparedData = {

    // find out all values of the each feature
    val landingValues = td.activity.map(_.activityId).distinct.collect
    val activityNameValues = td.activity.map(_.activityName).distinct.collect
    val userValues = td.activity.map(_.user).distinct.collect

    // map feature value to integer for each categorical feature
    val featureCategoricalIntMap = Map(
      "activityId" -> createCategoricalIntMap(landingValues, ""),
      "activityName" -> createCategoricalIntMap(activityNameValues, ""),
      "user" -> createCategoricalIntMap(userValues, "")
    )
    // index position of each feature in the vector
    val featureIndex = Map(
      "activityId" -> 0,
      "activityName" -> 1,
      "user" -> 2
    )

    // inject some default to cover default cases
    val defaults = Seq(
      new Activity(
        activityId = "",
        activityName = "",
        user = "",
        buy = false
      ),
     new Activity(
        activityId = "",
        activityName = "",
        user = "",
        buy = true
      ))

    val defaultRDD = sc.parallelize(defaults)
    val sessionRDD = td.activity.union(defaultRDD)

    val labeledPoints: RDD[LabeledPoint] = sessionRDD.map { activity =>
      logger.debug(s"${activity}")
      val label = if (activity.buy) 1.0 else 0.0

      val feature = new Array[Double](featureIndex.size)
      feature(featureIndex("activityId")) =
        featureCategoricalIntMap("activityId")(activity.activityId).toDouble
      feature(featureIndex("activityName")) =
        featureCategoricalIntMap("activityName")(activity.activityName).toDouble
      feature(featureIndex("user")) =
        featureCategoricalIntMap("user")(activity.user).toDouble
      LabeledPoint(label, Vectors.dense(feature))
    }.cache()

    logger.debug(s"labelelPoints count: ${labeledPoints.count()}")
    new PreparedData(
      labeledPoints = labeledPoints,
      featureIndex = featureIndex,
      featureCategoricalIntMap = featureCategoricalIntMap)
  }
}
