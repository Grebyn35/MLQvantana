package Objects;

import lombok.Data;

import java.util.ArrayList;

@Data
public class RSI {
    private double value;
    private int period;

    private String timeFormat;
    private String timestamp;

    private static double rma(ArrayList<Double> values, int period) {
        double sum = 0;
        for (int i = 0; i < period; i++) {
            sum += values.get(i);
        }
        double rma = sum / period;
        for (int i = period; i < values.size(); i++) {
            rma = (rma * (period - 1) + values.get(i)) / period;
        }
        return rma;
    }

    public static ArrayList<RSI> calculateRSI(ArrayList<Candlestick> candlesticks, int period) {
        ArrayList<Double> upValues = new ArrayList<>();
        ArrayList<Double> downValues = new ArrayList<>();
        ArrayList<RSI> rsis = new ArrayList<>();

        upValues.add(0.0);
        downValues.add(0.0);

        for (int i = 1; i < candlesticks.size(); i++) {
            double diff = candlesticks.get(i).getClose() - candlesticks.get(i - 1).getClose();
            upValues.add(Math.max(diff, 0));
            downValues.add(Math.max(-diff, 0));

            if (i >= period) {
                double rmaUp = rma(new ArrayList<>(upValues.subList(i - period + 1, i + 1)), period);
                double rmaDown = rma(new ArrayList<>(downValues.subList(i - period + 1, i + 1)), period);

                double rsi = rmaDown == 0 ? 100 : rmaUp == 0 ? 0 : 100 - (100 / (1 + rmaUp / rmaDown));

                RSI rsiO = new RSI();
                rsiO.setValue(rsi);
                rsiO.setPeriod(period);
                rsiO.setTimeFormat(candlesticks.get(i).getOpenTimeFormat());
                rsiO.setTimestamp(candlesticks.get(i).getOpenTime());

                rsis.add(rsiO);
            }
        }
        return rsis;
    }
}

