package com.b201.api.dto;

import java.time.LocalDateTime;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.jackson.Jacksonized;

@Getter
@Builder
@Jacksonized
@AllArgsConstructor
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
@JsonIgnoreProperties(ignoreUnknown = true)
public class AiResultDto {

	@NotNull(message = "capture_timestamp_utc는 필수입니다")
	private LocalDateTime captureTimestampUtc;

	@Valid
	@NotNull(message = "location 정보는 필수입니다")
	private Location location;

	@Valid
	@NotNull(message = "image_info는 필수입니다")
	private ImageInfo imageInfo;

	@Valid
	@NotEmpty(message = "detections 리스트는 비어있을 수 없습니다")
	private List<Detection> detections;

	@Getter
	@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
	public static class Location {
		@DecimalMin(value = "33.0", message = "위도는 33.0 이상이어야 합니다")
		@DecimalMax(value = "39.0", message = "위도는 39.0 이하이어야 합니다")
		private double latitude;

		@DecimalMin(value = "124.0", message = "경도는 124.0 이상이어야 합니다")
		@DecimalMax(value = "132.0", message = "경도는 132.0 이하이어야 합니다")
		private double longitude;

		@NotNull(message = "accuracy_meters는 필수입니다")
		private Double accuracyMeters;
	}

	@Getter
	@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
	public static class ImageInfo {
		@NotNull(message = "image_url은 필수입니다")
		private String imageUrl;

		@NotNull(message = "risk는 필수입니다")
		private Double risk;
	}

	@Getter
	@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
	public static class Detection {
		@NotNull(message = "category_name은 필수입니다")
		private String categoryName;
	}
}
