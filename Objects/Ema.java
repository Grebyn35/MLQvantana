package Objects;

import lombok.Data;

import java.util.ArrayList;

@Data
public class Ema {
    private ArrayList<Double> values; // Changed to an ArrayList to hold multiple values.
    private int length;

    private String timeFormat;
    private String timestamp;

    public static Ema calcEMA(ArrayList<Candlestick> candlesticks, int emaPeriod){
        ArrayList<Double> emaValues = new ArrayList<>();
        double smoothingFactor = 2.0 / (emaPeriod + 1.0);

        // Initialize the EMA with the first data point
        double previousEMA = candlesticks.get(0).getClose();
        emaValues.add(previousEMA);

        // Calculate the rest of the EMA values
        for (int i = 1; i < candlesticks.size(); i++) {
            double currentDataPoint = candlesticks.get(i).getClose();
            double currentEMA = (currentDataPoint - previousEMA) * smoothingFactor + previousEMA;
            emaValues.add(currentEMA);
            previousEMA = currentEMA;
        }
        Ema EMA = new Ema();
        EMA.setValues(emaValues);  // Set the whole ArrayList as the value.
        EMA.setLength(emaPeriod);
        EMA.setTimeFormat(candlesticks.get(candlesticks.size() - 1).getOpenTimeFormat());
        EMA.setTimestamp(candlesticks.get(candlesticks.size() - 1).getOpenTime());
        return EMA;
    }
}
