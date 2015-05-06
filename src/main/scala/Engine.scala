package main.scala

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(
    activityId: String,
    activityName: String,
    user: String
) extends Serializable

case class PredictedResult(
  score: Double
) extends Serializable

object LeadScoringEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("randomforest" -> classOf[Algorithm]),
      classOf[Serving])
  }
}