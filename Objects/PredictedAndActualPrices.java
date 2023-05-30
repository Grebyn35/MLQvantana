package Objects;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class PredictedAndActualPrices {

    private List<Double> actualPrices;
    private List<Double> predictedPrices;
    private List<Double> priceIntoFuture;
    private List<Double> portfolio;
}
