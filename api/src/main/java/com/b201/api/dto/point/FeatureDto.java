package com.b201.api.dto.point;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class FeatureDto {

	@Builder.Default
	private final String type = "Feature";
	private final GeometryDto geometry;
	private final PropertiesDto properties;
}