package com.b201.api.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class GeometryDto {
	
	@Builder.Default
	private final String type = "Point";
	private final double[] coordinates;
}
