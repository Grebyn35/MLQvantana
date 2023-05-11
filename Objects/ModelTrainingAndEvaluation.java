package Objects;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class ModelTrainingAndEvaluation {
    private List<Double> scores;
    private List<Double> trainingErrors;
    private List<Double> validationErrors;
}
