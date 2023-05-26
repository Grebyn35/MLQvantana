package Objects;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

@Data
public class Pair {
    private List<INDArray> featureList;
    private List<Candlestick> candlesticks;
    private List<INDArray> labelList;
}
