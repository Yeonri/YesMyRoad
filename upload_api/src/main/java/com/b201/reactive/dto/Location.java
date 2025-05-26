package com.b201.reactive.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Location {

    private double latitude;
    private double longitude;
    @JsonInclude(JsonInclude.Include.NON_NULL)  // null이면, JSON에 아예 안 나감
    private Double accuracyMeters;

}
