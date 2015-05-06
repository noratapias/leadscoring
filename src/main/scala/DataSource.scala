package main.scala

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.store.PEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import grizzled.slf4j.Logger

case class DataSourceParams(appName: String) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

 @transient lazy val logger = Logger[this.type]
  override
  def readTraining(sc: SparkContext): TrainingData = {

    // read all events of EVENT involving ENTITY_TYPE and TARGET_ENTITY_TYPE
    val viewPage: RDD[(String, Event)] = PEventStore.find(
      appName = dsp.appName,
      entityType = Some("user"),
      eventNames = Some(Seq("view")),
      targetEntityType = Some(Some("activity")))(sc)
      .map { event =>
        val activityName = try {
          event.properties.get[String]("activityName")
        } catch {
          case e: Exception => {
            logger.error(s"Cannot get activityName from event ${event}. ${e}.")
            throw e
          }
        }
        (activityName, event)
      }
     val buyItem: RDD[(String, Event)] = PEventStore.find(
      appName = dsp.appName,
      entityType = Some("user"),
      eventNames = Some(Seq("buy")),
      // targetEntityType is optional field of an event.
      targetEntityType = Some(Some("item")))(sc)
      // eventsDb.find() returns RDD[Event]
     .map { event =>
        val activityName = try {
          event.properties.get[String]("activityName")
        } catch {
          case e: Exception => {
            logger.error(s"Cannot get activityName from event ${event}. ${e}.")
            throw e
          }
        }
        (activityName, event)
      }
      val activity: RDD[Activity] = viewPage.cogroup(buyItem)
      .map { case (activityName, (viewIter, buyIter)) =>
        // the first view event of the session is the landing event
        val landing = viewIter.reduce{ (a, b) =>
          if (a.eventTime.isBefore(b.eventTime)) a else b
        }
        // any buy after landing
        val buy = buyIter.filter( b => b.eventTime.isAfter(landing.eventTime))
          .nonEmpty

        try {
          new Activity(
            activityId = landing.targetEntityId.get,
            activityName = landing.properties.getOrElse[String]("activityName", ""),
            user = landing.entityId,
            buy = buy
          )
        } catch {
          case e: Exception => {
            logger.error(s"Cannot create session data from ${landing}. ${e}.")
            throw e
          }
        }
      }.cache()
    new TrainingData(activity)
  }
}

case class Activity(
  activityId: String,
  activityName: String,
  user: String,
  buy: Boolean // buy or not
) extends Serializable

class TrainingData(
  val activity: RDD[Activity]
) extends Serializable
