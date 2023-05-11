package Objects;

import com.google.gson.annotations.SerializedName;
import lombok.Data;

import java.util.ArrayList;

@Data
public class CandlestickResponse {
    @SerializedName(value="retMsg")
    private String retMsg;
    @SerializedName(value="data")
    private ArrayList<Candlestick> data;
}
