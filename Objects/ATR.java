package Objects;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class ATR {
    private List<Double> values;  // Added field for the list of ATR values
    private double length;

    private String timeFormat;
    private String timestamp;

    public static ATR calculateATR(ArrayList<Candlestick> candles, int period) {
        ArrayList<Double> trueRanges = new ArrayList<>();
        List<Double> atrValues = new ArrayList<>();

        for (int i = 1; i < candles.size(); i++) {
            double high = candles.get(i).getHigh();
            double low = candles.get(i).getLow();
            double prevClose = candles.get(i - 1).getClose();

            double trueRange = Math.max(high - low, Math.max(Math.abs(high - prevClose), Math.abs(low - prevClose)));
            trueRanges.add(trueRange);
        }

        double sumTrueRanges = 0;
        for (int i = 0; i < period; i++) {
            sumTrueRanges += trueRanges.get(i);
        }

        double firstAtr = sumTrueRanges / period;
        atrValues.add(firstAtr);

        for (int i = period; i < trueRanges.size(); i++) {
            double prevAtr = atrValues.get(atrValues.size() - 1);
            double currentTrueRange = trueRanges.get(i);
            double atr = ((prevAtr * (period - 1)) + currentTrueRange) / period;
            atrValues.add(atr);
        }

        ATR atr = new ATR();
        atr.setValues(atrValues);  // Set the full list of ATR values
        atr.setLength(period);
        atr.setTimeFormat(candles.get(candles.size() - 1).getOpenTimeFormat());
        atr.setTimestamp(candles.get(candles.size() - 1).getOpenTime());
        return atr;
    }
}
