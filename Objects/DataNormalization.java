package Objects;

import kotlin.collections.ArrayDeque;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

@Data
public class DataNormalization {
    private List<INDArray> normalizedTrainFeatures;
    private List<INDArray> normalizedTrainLabels;
    private List<INDArray> normalizedTestFeatures;
    private List<INDArray> normalizedTestLabels;
    private List<INDArray> normalizedValidationFeatures;
    private List<INDArray> normalizedValidationLabels;

    double trainLabelMin;
    double trainLabelMax;
}
