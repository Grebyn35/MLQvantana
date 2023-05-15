import Objects.Candlestick;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class MACD {
    private double[] signalLine;
    private double[] macdLine;
    private double[] histogram;

    private double fastPeriod;
    private double slowPeriod;
    private double signalPeriod;

    private String[] timeFormat;
    private String timestamp;

    public static ArrayList<Candlestick> getLastNElements(List<Candlestick> originalArray, int n) {
        int startIndex = originalArray.size() - n;

        if (startIndex < 0) {
            startIndex = 0;
        }

        return new ArrayList<>(originalArray.subList(startIndex, originalArray.size()));
    }

    public static MACD calculateMACD(ArrayList<Candlestick> candles, int fastPeriod, int slowPeriod, int signalPeriod, int size) {
        candles = getLastNElements(candles, size);
        double[] emaFast = calculateEMA(candles, fastPeriod);
        double[] emaSlow = calculateEMA(candles, slowPeriod);

        // Calculate the MACD Line
        double[] macdLine = new double[candles.size()];
        for (int i = 0; i < candles.size(); i++) {
            macdLine[i] = emaFast[i] - emaSlow[i];
        }

        // Calculate the Signal Line
        double[] signalLine = calculateEMA(macdLine, signalPeriod);

        // Calculate the Histogram
        double[] histogram = new double[candles.size()];
        for (int i = 0; i < candles.size(); i++) {
            histogram[i] = macdLine[i] - signalLine[i];
        }

        String[] times = new String[candles.size()];
        for (int i = 0; i < candles.size(); i++) {
            times[i] = candles.get(i).getOpenTimeFormat();
        }

        MACD macd = new MACD();

        macd.setFastPeriod(fastPeriod);
        macd.setSlowPeriod(slowPeriod);
        macd.setSignalPeriod(signalPeriod);

        // Store the full MACD line, signal line, and histogram
        macd.setMacdLine(macdLine);
        macd.setSignalLine(signalLine);
        macd.setHistogram(histogram);

        macd.setTimeFormat(times);
        macd.setTimestamp(candles.get(candles.size() - 1).getOpenTime());

        return macd;
    }

    private static double[] calculateEMA(ArrayList<Candlestick> candles, int period) {
        double[] ema = new double[candles.size()];
        double multiplier = 2.0 / (period + 1);

        for (int i = 0; i < candles.size(); i++) {
            double closePrice = candles.get(i).getClose();

            if (i == 0) {
                ema[i] = closePrice;
            } else {
                ema[i] = (closePrice - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        return ema;
    }

    private static double[] calculateEMA(double[] values, int period) {
        double[] ema = new double[values.length];
        double multiplier = 2.0 / (period + 1);

        for (int i = 0; i < values.length; i++) {
            double value = values[i];

            if (i == 0) {
                ema[i] = value;
            } else {
                ema[i] = (value - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        return ema;
    }

}
