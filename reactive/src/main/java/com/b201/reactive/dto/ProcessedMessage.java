package com.b201.reactive.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ProcessedMessage {

    @JsonProperty("capture_timestamp_utc")
    private LocalDateTime captureTimestampUtc;
    private Location location;
    @JsonProperty("image_info")
    private ImageInfo imageInfo;
    private List<Detection> detections;

}
