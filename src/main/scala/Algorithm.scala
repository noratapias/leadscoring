package main.scala

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.linalg.Vectors

import grizzled.slf4j.Logger

case class RFAlgorithmParams(
  numTrees: Int,
  featureSubsetStrategy: String,
  impurity: String,
  maxDepth: Int,
  maxBins: Int,
  seed: Option[Int]
) extends Params

class RFModel(
  val forest: RandomForestModel,
  val featureIndex: Map[String, Int],
  val featureCategoricalIntMap: Map[String, Map[String, Int]]
) extends Serializable {
  override def toString = {
    s" forest: [${forest}]" +
    s" featureIndex: ${featureIndex}" +
    s" featureCategoricalIntMap: ${featureCategoricalIntMap}"
  }
}

class Algorithm(val ap: RFAlgorithmParams)
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, RFModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, pd: PreparedData): RFModel = {

    val categoricalFeaturesInfo = pd.featureCategoricalIntMap
      .map { case (f, m) =>
        (pd.featureIndex(f), m.size)
      }

    logger.info(s"categoricalFeaturesInfo: ${categoricalFeaturesInfo}")

    // use random seed if seed is not specified
    val seed = ap.seed.getOrElse(scala.util.Random.nextInt())

    val forestModel: RandomForestModel = RandomForest.trainRegressor(
      input = pd.labeledPoints,
      categoricalFeaturesInfo = categoricalFeaturesInfo,
      numTrees = ap.numTrees,
      featureSubsetStrategy = ap.featureSubsetStrategy,
      impurity = ap.impurity,
      maxDepth = ap.maxDepth,
      maxBins = ap.maxBins,
      seed = seed)

    new RFModel(
      forest = forestModel,
      featureIndex = pd.featureIndex,
      featureCategoricalIntMap = pd.featureCategoricalIntMap
    )
  }

  def predict(model: RFModel, query: Query): PredictedResult = {

    val featureIndex = model.featureIndex
    val featureCategoricalIntMap = model.featureCategoricalIntMap

    val activityId = query.activityId
    val activityName = query.activityName
    val user = query.user

    // look up categorical feature Int for landingPageId
    val actFeature = lookupCategoricalInt(
      featureCategoricalIntMap = featureCategoricalIntMap,
      feature = "activityId",
      value = activityId,
      default = ""
    ).toDouble


    // look up categorical feature Int for referrerId
    val actNameFeature = lookupCategoricalInt(
      featureCategoricalIntMap = featureCategoricalIntMap,
      feature = "activityName",
      value = activityName,
      default = ""
    ).toDouble

    // look up categorical feature Int for brwoser
    val userFeature = lookupCategoricalInt(
      featureCategoricalIntMap = featureCategoricalIntMap,
      feature = "user",
      value = user,
      default = ""
    ).toDouble

    // create feature Array
    val feature = new Array[Double](model.featureIndex.size)
    feature(featureIndex("activityId")) = actFeature
    feature(featureIndex("activityName")) = actNameFeature
    feature(featureIndex("user")) = userFeature

    val score = model.forest.predict(Vectors.dense(feature))
    new PredictedResult(score)
  }

  private def lookupCategoricalInt(
    featureCategoricalIntMap: Map[String, Map[String, Int]],
    feature: String,
    value: String,
    default: String): Int = {

    featureCategoricalIntMap(feature)
      .get(value)
      .getOrElse {
        logger.info(s"Unknown ${feature} ${value}." +
          " Default feature value will be used.")
        // use default feature value
        featureCategoricalIntMap(feature)(default)
      }
  }
}
