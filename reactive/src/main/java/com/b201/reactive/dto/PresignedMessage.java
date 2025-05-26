package com.b201.reactive.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class PresignedMessage {

    @JsonProperty("capture_timestamp_utc")
    private LocalDateTime captureTimestampUtc;
    private Location location;
    @JsonProperty("image_info")
    private ImageInfo imageInfo;
}
