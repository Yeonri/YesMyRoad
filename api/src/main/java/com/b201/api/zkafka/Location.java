package com.b201.api.zkafka;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Location {

    private double latitude;
    private double longitude;
    @JsonInclude(JsonInclude.Include.NON_NULL)  // null이면, JSON에 아예 안 나감
    private Double accuracyMeters;

}
