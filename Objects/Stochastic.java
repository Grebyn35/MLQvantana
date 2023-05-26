package Objects;

import lombok.Data;

import java.util.ArrayList;

@Data
public class Stochastic {
    private double percentK;
    private double percentD;

    private int kPeriodLength;
    private int dPeriodLength;

    private String timeFormat;
    private String timestamp;

    public static ArrayList<Stochastic> calculateStochastic(ArrayList<Candlestick> candlesticks, int kPeriod, int dPeriod) {
        ArrayList<Stochastic> stochastics = new ArrayList<>();

        int size = candlesticks.size();
        if (size < kPeriod || kPeriod <= 0 || dPeriod <= 0) {
            Stochastic stochastic = new Stochastic();
            stochastic.setPercentK(0);
            stochastic.setPercentD(0);
            stochastic.setKPeriodLength(kPeriod);
            stochastic.setDPeriodLength(dPeriod);
            stochastic.setTimeFormat(candlesticks.get(size - 1).getOpenTimeFormat());
            stochastic.setTimestamp(candlesticks.get(size - 1).getOpenTime());
            stochastics.add(stochastic);
            return stochastics;
        }

        ArrayList<Double> percentKValues = new ArrayList<>();
        ArrayList<Double> percentDValues = new ArrayList<>();

        for (int i = kPeriod - 1; i < size; i++) {
            double highestHigh = Double.NEGATIVE_INFINITY;
            double lowestLow = Double.POSITIVE_INFINITY;
            for (int j = i - kPeriod + 1; j <= i; j++) {
                Candlestick candle = candlesticks.get(j);
                highestHigh = Math.max(highestHigh, candle.getHigh());
                lowestLow = Math.min(lowestLow, candle.getLow());
            }
            double close = candlesticks.get(i).getClose();
            double percentK = 100 * (close - lowestLow) / (highestHigh - lowestLow);
            percentKValues.add(percentK);
        }

        for (int i = dPeriod - 1; i < percentKValues.size(); i++) {
            double sumPercentK = 0;
            for (int j = i - dPeriod + 1; j <= i; j++) {
                sumPercentK += percentKValues.get(j);
            }
            double percentD = sumPercentK / dPeriod;
            percentDValues.add(percentD);

            Stochastic stochastic = new Stochastic();
            stochastic.setPercentK(percentKValues.get(i));
            stochastic.setPercentD(percentD);
            stochastic.setKPeriodLength(kPeriod);
            stochastic.setDPeriodLength(dPeriod);
            stochastic.setTimeFormat(candlesticks.get(i).getOpenTimeFormat());
            stochastic.setTimestamp(candlesticks.get(i).getOpenTime());
            stochastics.add(stochastic);
        }

        return stochastics;
    }

}