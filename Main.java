import Objects.*;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.opencv.ml.EM;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    static int nEpochs = 25;
    static double dropout = 0.00;
    static int params = 6;

    static double portfolio = 2000;

    public static void main(String[] args) throws IOException {
        //Initialize the hyperparameter tuning values
        ArrayList<Integer> lookbackList = new ArrayList<>();
        ArrayList<Integer> stepsIntoFutureList = new ArrayList<>();

        //Add lookback parameters for training
        //Default is 6
        lookbackList.add(30);
        //lookbackList.add(14);
        //lookbackList.add(16);
        //lookbackList.add(18);
        //lookbackList.add(20);

        //Add stepsIntoFuture parameters for training
        //Default is 3
        //stepsIntoFutureList.add(1);
        //stepsIntoFutureList.add(2);
        stepsIntoFutureList.add(4);
        //stepsIntoFutureList.add(4);
        //stepsIntoFutureList.add(5);
        //stepsIntoFutureList.add(6);

        ArrayList<Candlestick> candlesticks = returnCandlestickList("bybit", "ethusdt", "30m", "usdt-perpetual", 40000, "2020-01-01%2000:00:00");
        trainModel(candlesticks, lookbackList, stepsIntoFutureList);
    }
    public static void trainModel(ArrayList<Candlestick> candlesticks, ArrayList<Integer> lookbackList, ArrayList<Integer> stepsIntoFutureList){

        for(int i = 0; i<lookbackList.size();i++){
            for(int j = 0; j<stepsIntoFutureList.size();j++){
                System.out.println(i + "/" + lookbackList.size() + ", " + j + "/" + stepsIntoFutureList.size());
                // Step 1: Create the features and labels
                Pair pair = createFeaturesAndLabels(candlesticks, lookbackList.get(i), stepsIntoFutureList.get(j));
                List<INDArray> featureList = pair.getFeatureList();
                List<INDArray> labelList = pair.getLabelList();
                List<Candlestick> candlestickList = pair.getCandlesticks();

                // Step 2: Split the data into train, test and validation sets
                DataSplit dataSplit = splitData(featureList, labelList, candlestickList);
                List<INDArray> trainFeatures = dataSplit.getTrainFeatures();
                List<INDArray> trainLabels = dataSplit.getTrainLabels();

                List<INDArray> testFeatures = dataSplit.getTestFeatures();
                List<INDArray> testLabels = dataSplit.getTestLabels();

                List<INDArray> validationFeatures = dataSplit.getValidationFeatures();
                List<INDArray> validationLabels = dataSplit.getValidationLabels();

                // Step 3: Normalize data
                DataNormalization dataNormalization = normalizeData(trainFeatures, trainLabels, testFeatures, testLabels, validationFeatures, validationLabels);
                List<INDArray> normalizedTrainFeatures = dataNormalization.getNormalizedTrainFeatures();
                List<INDArray> normalizedTrainLabels = dataNormalization.getNormalizedTrainLabels();

                List<INDArray> normalizedTestFeatures = dataNormalization.getNormalizedTestFeatures();
                List<INDArray> normalizedTestLabels = dataNormalization.getNormalizedTestLabels();

                List<INDArray> normalizedValidationFeatures = dataNormalization.getNormalizedValidationFeatures();
                List<INDArray> normalizedValidationLabels = dataNormalization.getNormalizedValidationLabels();


                // Step 4: Initialize the learning model being used
                MultiLayerNetwork model = returnModel();
                model.init();

                // Step 5: Train the model and evaluate
                ModelTrainingAndEvaluation modelTrainingAndEvaluation = trainAndEvaluateModel(model, normalizedTrainFeatures, normalizedTrainLabels, normalizedValidationFeatures, normalizedValidationLabels, dataSplit.getTrainSize(), lookbackList.get(i));
                List<Double> scores = modelTrainingAndEvaluation.getScores();
                List<Double> trainingErrors = modelTrainingAndEvaluation.getTrainingErrors();
                List<Double> validationErrors = modelTrainingAndEvaluation.getValidationErrors();

                //Save model to file

                // Step 6: Predict the test data and rescale
                PredictedAndActualPrices predictedAndActualPrices = predictAndRescalePrices(model, normalizedTestFeatures, normalizedTestLabels, testLabels, testFeatures, lookbackList.get(i), stepsIntoFutureList.get(j), dataSplit.getCandlesticks());
                List<Double> actualPrices = predictedAndActualPrices.getActualPrices();
                List<Double> predictedPrices = predictedAndActualPrices.getPredictedPrices();
                List<Double> portfolio = predictedAndActualPrices.getPortfolio();
                List<Double> priceIntoFuture = predictedAndActualPrices.getPriceIntoFuture();

                // Step 7: Show the results
                showResults(actualPrices, predictedPrices, scores, trainingErrors, validationErrors, portfolio, priceIntoFuture);
            }
        }
    }
    public static void showResults(List<Double> actualPrices, List<Double> predictedPrices, List<Double> scores, List<Double> trainingErrors, List<Double> validationErrors, List<Double> portfolio, List<Double> priceIntoFuture){
        plotPredictions(actualPrices, predictedPrices);
        plotPortfolio(portfolio);
        plotCandlesticsIntoFuture(priceIntoFuture);
        //plotScore(scores);
        plotTrainingErrors(trainingErrors);
        plotValidationErrors(validationErrors);
        /*System.out.println("Actual Price: " + actualPrices);
        System.out.println("Predicted Price: " + predictedPrices);
        System.out.println("Learning Curve: " + scores);
        System.out.println("Training Mean Squared Errors: " + trainingErrors);
        System.out.println("Validation Mean Squared Errors: " + validationErrors);*/
    }
    private static void plotTrainingErrors(List<Double> trainingErrors) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Training errors")
                .xAxisTitle("Epoch")
                .yAxisTitle("Score")
                .build();

        // Add data to the chart
        chart.addSeries("Training errors", trainingErrors);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }

    private static void plotValidationErrors(List<Double> validationErrors) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Validation errors")
                .xAxisTitle("Epoch")
                .yAxisTitle("Score")
                .build();

        // Add data to the chart
        chart.addSeries("Validation errors", validationErrors);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotScore(List<Double> scores) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Learning Curve")
                .xAxisTitle("Epoch")
                .yAxisTitle("Score")
                .build();

        // Add data to the chart
        chart.addSeries("Score", scores);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotCandlesticsIntoFuture(List<Double> priceIntoFuture) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Future close")
                .xAxisTitle("Timeseries")
                .yAxisTitle("Value")
                .build();

        // Add data to the chart
        chart.addSeries("Future close", priceIntoFuture);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotPortfolio(List<Double> portfolio) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Portfolio")
                .xAxisTitle("Timeseries")
                .yAxisTitle("Value")
                .build();

        // Add data to the chart
        chart.addSeries("Portfolio", portfolio);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    private static void plotPredictions(List<Double> actual, List<Double> predicted) {
        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Predicted vs. Actual Prices")
                .xAxisTitle("Time")
                .yAxisTitle("Price")
                .build();

        // Add data to the chart
        chart.addSeries("Actual", actual);
        chart.addSeries("Predicted", predicted);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    public static PredictedAndActualPrices predictAndRescalePrices(MultiLayerNetwork model, List<INDArray> normalizedTestFeatures, List<INDArray> normalizedTestLabels, List<INDArray> testLabels, List<INDArray> testFeatures, int lookback, int stepsIntoFuture, List<Candlestick> candlestickList){
        List<Double> predictedPrices = new ArrayList<>();
        List<Double> actualPrices = new ArrayList<>();
        List<Double> portfolioList = new ArrayList<>();
        List<Double> priceIntoFutureList = new ArrayList<>();
        Evaluation eval = new Evaluation(2);
        double wins = 0;
        double losses = 0;
        //Evaluate the model
        for (int i = 0; i < normalizedTestFeatures.size(); i++) {
            INDArray predicted = model.output(normalizedTestFeatures.get(i).reshape(1, params, lookback));
            eval.eval(normalizedTestLabels.get(i).reshape(1, 1, lookback), predicted);
            //To simulate trading
            if(i<normalizedTestFeatures.size()-stepsIntoFuture){
                System.out.println("(current close: " + candlestickList.get(i).getClose() + " | future close: "  + candlestickList.get(i+stepsIntoFuture).getClose() + "). label: " + testLabels.get(i).getDouble(0) + ". predicted: " + predicted.getDouble(0));
                //double fees = (candlestickList.get(i+stepsIntoFuture).getClose() * 0.0006) + (candlestickList.get(i).getClose() * 0.0006);
                double fees = 0;
                if(predicted.getDouble(0) > 0.5){
                    if(candlestickList.get(i+stepsIntoFuture).getClose() > candlestickList.get(i).getClose()){
                        wins++;
                    }
                    else{
                        losses++;
                    }
                    double profitAfterFees = (candlestickList.get(i+stepsIntoFuture).getClose() - candlestickList.get(i).getClose()) - fees;
                    portfolio+=profitAfterFees;
                    System.out.println("predicted: " + predicted.getDouble(0) + ". Profit after fees: " + profitAfterFees);
                }
                else if(predicted.getDouble(0) < 0.5){
                    if(candlestickList.get(i).getClose() > candlestickList.get(i+stepsIntoFuture).getClose()){
                        wins++;
                    }
                    else{
                        losses++;
                    }
                    double profitAfterFees = (candlestickList.get(i).getClose() - candlestickList.get(i+stepsIntoFuture).getClose()) - fees;
                    portfolio+=profitAfterFees;
                    System.out.println("predicted: " + predicted.getDouble(0) + ". Profit after fees: " + profitAfterFees);
                }
                //Add portfolio value to a list to later see growth
                portfolioList.add(portfolio);
                priceIntoFutureList.add(candlestickList.get(i+stepsIntoFuture).getClose());
            }
            // Add data to plot
            predictedPrices.add(predicted.getDouble(0));
            actualPrices.add(testLabels.get(i).getDouble(0));
        }
        System.out.println("wins: " + wins);
        System.out.println("losses: " + losses);
        PredictedAndActualPrices predictedAndActualPrices = new PredictedAndActualPrices();
        predictedAndActualPrices.setPredictedPrices(predictedPrices);
        predictedAndActualPrices.setPortfolio(portfolioList);
        predictedAndActualPrices.setActualPrices(actualPrices);
        double accuracy = eval.accuracy();
        double precision = eval.precision();
        double recall = eval.recall();
        double f1 = eval.f1();

        System.out.println("Lookback Hyperparameter: " + lookback);
        System.out.println("StepsIntoFuture Hyperparameter: " + stepsIntoFuture);
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F1 Score: " + f1);
        return predictedAndActualPrices;
    }
    public static ModelTrainingAndEvaluation trainAndEvaluateModel(MultiLayerNetwork model, List<INDArray> normalizedTrainFeatures, List<INDArray> normalizedTrainLabels, List<INDArray> normalizedValidationFeatures, List<INDArray> normalizedValidationLabels, int trainSize, int lookback){
        List<Double> scores = new ArrayList<>();
        //Train the model
        int minibatchSize = 32; // You can tweak this value

        List<Double> trainingErrors = new ArrayList<>();
        List<Double> validationErrors = new ArrayList<>();
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            // reset error for this epoch
            double trainingError = 0;
            for (int i = 0; i < trainSize; i += minibatchSize) {
                List<INDArray> featuresMinibatch = new ArrayList<>();
                List<INDArray> labelsMinibatch = new ArrayList<>();
                for (int j = 0; j < minibatchSize && i + j < trainSize; j++) {
                    featuresMinibatch.add(normalizedTrainFeatures.get(i + j).reshape(1, params, lookback));
                    labelsMinibatch.add(normalizedTrainLabels.get(i + j).reshape(1, 1, lookback));  // No reshaping
                }
                model.fit(Nd4j.concat(0, featuresMinibatch.toArray(new INDArray[0])), Nd4j.concat(0, labelsMinibatch.toArray(new INDArray[0])));
                trainingError += model.score();
            }
            scores.add(model.score());
            trainingError /= (trainSize / minibatchSize);
            double validationError = 0;
            Evaluation validationEval = new Evaluation(2);  // Changed to Evaluation
            for (int i = 0; i < normalizedValidationFeatures.size(); i++) {
                INDArray predicted = model.output(normalizedValidationFeatures.get(i).reshape(1, params, lookback)); // Updated to reflect the lookback period
                validationEval.eval(normalizedValidationLabels.get(i).reshape(1,1,lookback), predicted);  // No reshaping
            }
            validationError = 1 - validationEval.accuracy();  // Calculate error as 1 - accuracy
            System.out.println("Epoch " + epoch + " / " + nEpochs + " : training error = " + trainingError + ", validation error = " + validationError);

            validationErrors.add(validationError);
            trainingErrors.add(trainingError);
        }

        ModelTrainingAndEvaluation modelTrainingAndEvaluation = new ModelTrainingAndEvaluation();
        modelTrainingAndEvaluation.setScores(scores);
        modelTrainingAndEvaluation.setTrainingErrors(trainingErrors);
        modelTrainingAndEvaluation.setValidationErrors(validationErrors);

        return modelTrainingAndEvaluation;
    }
    public static DataNormalization normalizeData(List<INDArray> trainFeatures, List<INDArray> trainLabels, List<INDArray> testFeatures, List<INDArray> testLabels, List<INDArray> validationFeatures, List<INDArray> validationLabels){
        // Calculate min and max only from training set
        INDArray trainMin = trainFeatures.get(0).dup();
        INDArray trainMax = trainFeatures.get(0).dup();

        for (INDArray features : trainFeatures) {
            trainMin = Transforms.min(trainMin, features);
            trainMax = Transforms.max(trainMax, features);
        }

        // Normalize train, test, and validation sets using trainMin and trainMax
        List<INDArray> normalizedTrainFeatures = normalizeFeatures(trainFeatures, trainMin, trainMax);
        List<INDArray> normalizedTestFeatures = normalizeFeatures(testFeatures, trainMin, trainMax);
        List<INDArray> normalizedValidationFeatures = normalizeFeatures(validationFeatures, trainMin, trainMax);

        // Normalize labels
        double trainLabelMin = trainLabels.stream().mapToDouble(INDArray::getDouble).min().orElse(0);
        double trainLabelMax = trainLabels.stream().mapToDouble(INDArray::getDouble).max().orElse(1);

        List<INDArray> normalizedTrainLabels = returnNormalizedLabels(trainLabels, trainLabelMin, trainLabelMax);
        List<INDArray> normalizedTestLabels = returnNormalizedLabels(testLabels, trainLabelMin, trainLabelMax);
        List<INDArray> normalizedValidationLabels = returnNormalizedLabels(validationLabels, trainLabelMin, trainLabelMax);

        DataNormalization dataNormalization = new DataNormalization();
        dataNormalization.setNormalizedTestFeatures(normalizedTestFeatures);
        dataNormalization.setNormalizedTestLabels(normalizedTestLabels);

        dataNormalization.setNormalizedTrainFeatures(normalizedTrainFeatures);
        dataNormalization.setNormalizedTrainLabels(normalizedTrainLabels);

        dataNormalization.setNormalizedValidationFeatures(normalizedValidationFeatures);
        dataNormalization.setNormalizedValidationLabels(normalizedValidationLabels);

        dataNormalization.setTrainLabelMin(trainLabelMin);
        dataNormalization.setTrainLabelMax(trainLabelMax);
        return dataNormalization;
    }
    public static DataSplit splitData(List<INDArray> featureList, List<INDArray> labelList, List<Candlestick> candlesticks){
        int trainSize = (int) (0.7 * featureList.size());
        int testSize = (int) (0.15 * featureList.size());
        int candlestickSize = (int) (0.15 * candlesticks.size());

        List<INDArray> trainFeatures = featureList.subList(0, trainSize);
        List<INDArray> trainLabels = labelList.subList(0, trainSize);

        List<Candlestick> testCandlesticks = candlesticks.subList(trainSize, trainSize + candlestickSize);

        List<INDArray> testFeatures = featureList.subList(trainSize, trainSize + testSize);
        List<INDArray> testLabels = labelList.subList(trainSize, trainSize + testSize);

        List<INDArray> validationFeatures = featureList.subList(trainSize + testSize, featureList.size());
        List<INDArray> validationLabels = labelList.subList(trainSize + testSize, labelList.size());

        DataSplit dataSplit = new DataSplit();

        dataSplit.setTrainFeatures(trainFeatures);
        dataSplit.setTrainLabels(trainLabels);

        dataSplit.setCandlesticks(testCandlesticks);

        dataSplit.setTestFeatures(testFeatures);
        dataSplit.setTestLabels(testLabels);

        dataSplit.setValidationFeatures(validationFeatures);
        dataSplit.setValidationLabels(validationLabels);

        dataSplit.setTrainSize(trainSize);
        dataSplit.setTestSize(testSize);
        return dataSplit;
    }
    public static Pair createFeaturesAndLabels(ArrayList<Candlestick> candlesticks, int lookback, int stepsIntoFuture) {
        List<INDArray> featureList = new ArrayList<>();
        List<INDArray> labelList = new ArrayList<>();
        List<Candlestick> candlestickList = new ArrayList<>();

        MACD macd = MACD.calculateMACD(new ArrayList<>(candlesticks), 12, 26, 9, candlesticks.size());

        double[] macdLines = macd.getMacdLine();
        double[] signalLines = macd.getSignalLine();
        double[] histograms = macd.getHistogram();
        Ema ema200 = Ema.calcEMA(candlesticks, 200);
        Ema ema20 = Ema.calcEMA(candlesticks, 20);
        System.out.println("loaded a dataset size of " + candlesticks.size());

        //Create the features and labels
        for (int j = lookback; j < candlesticks.size() - stepsIntoFuture; j++) {
            // Features now have lookback * 5 size because for each lookback step we have 5 values (open, high, low, close, volume)
            INDArray features = Nd4j.create(new double[lookback * params]);
            for (int i = 0; i < lookback; i++) {
                if (j + stepsIntoFuture < candlesticks.size()) {
                    double currentClosePrice = candlesticks.get(j - i).getClose();
                    double previousClosePrice = (j - i - 1 >= 0) ? candlesticks.get(j - i - 1).getClose() : 0;

                    double closePriceDifference = currentClosePrice - previousClosePrice; // Considering fees at buying and selling

                    features.putScalar(i * params + 0, closePriceDifference);
                    features.putScalar(i * params + 1, macdLines[j - i]);
                    features.putScalar(i * params + 2, signalLines[j - i]);
                    features.putScalar(i * params + 3, histograms[j - i]);
                    features.putScalar(i * params + 4, ema200.getValues().get(j - i));
                    features.putScalar(i * params + 5, ema20.getValues().get(j - i));

                }
            }

            // Labels now have lookback size because for each lookback step we have 1 value (up or down)
            INDArray labels = Nd4j.create(new double[lookback]);
            for (int i = 0; i < lookback; i++) {
                if (j + stepsIntoFuture < candlesticks.size()) {

                    double currentClosePrice = candlesticks.get(j - i).getClose();
                    double futureClosePrice = candlesticks.get(j + stepsIntoFuture - i).getClose();

                    labels.putScalar(i, futureClosePrice > currentClosePrice ? 1 : 0);
                }
            }
            candlestickList.add(candlesticks.get(j));
            featureList.add(features);
            labelList.add(labels);
        }
        Pair pair = new Pair();
        pair.setCandlesticks(candlestickList);
        pair.setFeatureList(featureList);
        pair.setLabelList(labelList);

        return pair;
    }

    public static MultiLayerNetwork returnModel() {
        //Create the model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam())
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(params)  // Adjusted to account for lookback
                        .nOut(300)
                        .activation(Activation.TANH)
                        .dropOut(dropout)
                        .build())
                .layer(1, new LSTM.Builder()
                        .nIn(300)
                        .nOut(600)
                        .activation(Activation.TANH)
                        .dropOut(dropout)
                        .build())
                .layer(2, new LSTM.Builder()
                        .nIn(600)
                        .nOut(300)
                        .activation(Activation.TANH)
                        .dropOut(dropout)
                        .build())
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(300)
                        .nOut(1)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        int listenerFrequency = 10;  // Adjust this to control how often the listener will receive updates
        model.setListeners(new ScoreIterationListener(listenerFrequency));
        return model;
    }
    private static List<INDArray> returnNormalizedLabels(List<INDArray> labels, double min, double max) {
        List<INDArray> normalizedLabels = new ArrayList<>();
        for (INDArray label : labels) {
            INDArray normalizedLabel = label.sub(min).div(max - min);
            normalizedLabels.add(normalizedLabel);
        }
        return normalizedLabels;
    }
    // Method for feature normalization
    private static List<INDArray> normalizeFeatures(List<INDArray> features, INDArray min, INDArray max) {
        List<INDArray> normalizedFeatures = new ArrayList<>();
        for (INDArray feature : features) {
            INDArray normalizedFeature = feature.sub(min).div(max.sub(min));
            normalizedFeatures.add(normalizedFeature);
        }
        return normalizedFeatures;
    }
    public static ArrayList<Candlestick> returnCandlestickList(String exchange, String symbol, String interval, String market, int limit, String from) throws IOException {
        ArrayList<Candlestick> candlesticks = new ArrayList<>();
        String endpoint;
        int iterations = 1;
        if(limit>5000){
            iterations = (int) Math.ceil(limit / 5000.0);
        }
        for(int i = 0; i<iterations;i++){
            double internalLimit;
            if(limit>5000){
                internalLimit = 5000;
            }
            else{
                internalLimit = limit;
            }
            if(i==0){
                endpoint = "https://qvantana.herokuapp.com/kline?symbol=" + symbol + "&interval=" + interval + "&exchange=" + exchange + "&market=" + market + "&limit=" + (int)internalLimit + "&from=" + from;
            }
            else{
                endpoint = "https://qvantana.herokuapp.com/kline?symbol=" + symbol + "&interval=" + interval + "&exchange=" + exchange + "&market=" + market + "&limit=" + (int)internalLimit + "&from=" + candlesticks.get(candlesticks.size()-1).getOpenTime();
            }
            System.out.println(endpoint);
            Document doc = Jsoup.connect(endpoint)
                    .ignoreContentType(true)
                    .timeout(0)
                    .get();

            JsonElement json = JsonParser.parseString(doc.text());
            Gson g = new Gson();
            JsonElement jsonArrayCandlesticks =  json.getAsJsonObject().get("data");
            for (JsonElement candlestickItem : jsonArrayCandlesticks.getAsJsonArray()) {
                Candlestick candlestick = g.fromJson(candlestickItem, Candlestick.class);
                if (!candlesticks.contains(candlestick)) {
                    candlesticks.add(candlestick);
                }
            }
            limit -= jsonArrayCandlesticks.getAsJsonArray().size();
        }
        return candlesticks;
    }
}
