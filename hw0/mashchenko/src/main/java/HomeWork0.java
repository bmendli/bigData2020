import com.github.davidmoten.geo.GeoHash;
import com.github.davidmoten.geo.LatLong;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.expressions.Window;
import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.corr;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.first;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.udf;
import static org.apache.spark.sql.functions.variance;
import org.apache.spark.sql.types.DataTypes;

public class HomeWork0 {

    private static SparkSession sparkSession;

    public static void main(String[] args) {
        sparkSession = SparkSession
                .builder()
                .appName("Spakl hw 0")
                .config("spark.master", "local")
                .getOrCreate();

        final Dataset<Row> dataset = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("mode", "DROPMALFORMED")
                .option("escape", "\"")
                .csv("/Users/bogdan.mashchenko/github/technopolis/bigData2020/hw0/mashchenko/src/main/resources/AB_NYC_2019.csv");

        // median
        dataset.groupBy("room_type")
                .agg(callUDF("percentile_approx", col("price"), lit(0.5)).as("median"))
                .show();
        // +---------------+------+
        // |      room_type|median|
        // +---------------+------+
        // |    Shared room|  45.0|
        // |Entire home/apt| 160.0|
        // |   Private room|  70.0|
        // +---------------+------+

        //average
        dataset.groupBy("room_type")
                .agg(avg("price").as("avg"))
                .show();
        // +---------------+------------------+
        // |      room_type|               avg|
        // +---------------+------------------+
        // |    Shared room| 70.13298791018998|
        // |Entire home/apt|211.88216032823104|
        // |   Private room| 89.51396823968689|
        // +---------------+------------------+

        // variance
        dataset.groupBy("room_type")
                .agg(variance("price").as("variance"))
                .show();
        // +---------------+------------------+
        // |      room_type|          variance|
        // +---------------+------------------+
        // |    Shared room|10365.890682680929|
        // |Entire home/apt| 80852.24645965557|
        // |   Private room|23907.680804069663|
        // +---------------+------------------+

        // mode
        dataset.withColumn("count", count("price").over(Window.partitionBy("room_type")))
                .withColumn("price_mode", first("price").over(Window.orderBy("count").partitionBy("room_type")).as("mode"))
                .groupBy("room_type")
                .agg(first("price_mode").as("mode"))
                .show();
        // +---------------+----+
        // |      room_type|mode|
        // +---------------+----+
        // |    Shared room|  40|
        // |Entire home/apt| 225|
        // |   Private room| 149|
        // +---------------+----+

        // min price
        dataset.orderBy("price")
                .show(1);
        // +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
        // |      id|                name|host_id|host_name|neighbourhood_group|     neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
        // +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
        // |18750597|Huge Brooklyn Bro...|8993084| Kimberly|           Brooklyn|Bedford-Stuyvesant|40.69023|-73.95428|Private room|    0|             4|                1| 2018-01-06|             0.05|                           4.0|              28|
        // +--------+--------------------+-------+---------+-------------------+------------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+

        // max price
        dataset.orderBy(desc("price"))
                .show(1);
        // +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
        // |     id|                name|host_id|host_name|neighbourhood_group|  neighbourhood|latitude|longitude|   room_type|price|minimum_nights|number_of_reviews|last_review|reviews_per_month|calculated_host_listings_count|availability_365|
        // +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+
        // |9528920|Quiet, Clean, Lit...|3906464|      Amy|          Manhattan|Lower East Side|40.71355|-73.98507|Private room| 9999|            99|                6| 2016-01-01|             0.14|                           1.0|              83|
        // +-------+--------------------+-------+---------+-------------------+---------------+--------+---------+------------+-----+--------------+-----------------+-----------+-----------------+------------------------------+----------------+

        // correlation
        dataset.agg(
                corr("price", "minimum_nights").as("correlation_between_price_and_minimum_nights"),
                corr("price", "number_of_reviews").as("correlation_between_price_and_number_of_reviews")
        ).show();
        // +--------------------------------------------+-----------------------------------------------+
        // |correlation_between_price_and_minimum_nights|correlation_between_price_and_number_of_reviews|
        // +--------------------------------------------+-----------------------------------------------+
        // |                         0.04238800501413225|                           -0.04806955416645...|
        // +--------------------------------------------+-----------------------------------------------+


        // max price for square 5km x 5km
        UserDefinedFunction geoHash = udf((UDF3<Double, Double, Integer, String>) GeoHash::encodeHash, DataTypes.StringType);
        sparkSession.sqlContext().udf().register("geoHash", geoHash);
        final Row result = dataset.withColumn("geoHash", geoHash.apply(col("latitude").cast(DataTypes.DoubleType), col("longitude").cast(DataTypes.DoubleType), lit(5)))
                .withColumn("price", col("price").cast(DataTypes.LongType))
                .groupBy("geoHash")
                .mean("price")
                .orderBy(col("avg(price)").desc())
                .first();

        final LatLong latLong = GeoHash.decodeHash(result.getString(0));
        System.out.printf("latitude : longitude = %f : %f, price = %f%n", latLong.getLat(), latLong.getLon(), result.getDouble(1));
        // latitude : longitude = 40,583496 : -73,718262, price = 350,000000
    }
}