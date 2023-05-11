package Objects;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

@Data
public class DataSplit {
     private List<INDArray> trainFeatures;
     private List<INDArray> trainLabels;


     private List<INDArray> testFeatures;
     private List<INDArray> testLabels;

     private List<INDArray> validationFeatures;
     private List<INDArray> validationLabels;
     private int trainSize;
     private int testSize;
}
