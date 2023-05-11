package Objects;

import com.google.gson.annotations.SerializedName;
import lombok.Data;

@Data
public class Candlestick {
    @SerializedName(value="symbol")
    private String symbol;
    @SerializedName(value="period")
    private String period;

    @SerializedName(value="startAt")
    private String startAt;
    @SerializedName(value="volume")
    private double volume;
    @SerializedName(value="open")
    private double open;
    @SerializedName(value="high")
    private double high;
    @SerializedName(value="low")
    private double low;
    @SerializedName(value="close")
    private double close;
    @SerializedName(value="interval")
    private String timeInterval;
    @SerializedName(value="openTime")
    private String openTime;
    @SerializedName(value="turnover")
    private String turnover;
    private String openTimeFormat;
}
