package com.b201.reactive.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ImageInfo {

    @JsonProperty("uuid")
    private String uuid;

    @JsonProperty("presigned_url")
    private String presignedUrl;

    @JsonProperty("image_url")
    private String imageUrl;

    private Double risk;

}