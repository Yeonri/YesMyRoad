package com.b201.api.dto;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * 손상 데이터 좌표값들을 geoJson으로 보내기 위한 Dto
 */
@Getter
@Builder
@AllArgsConstructor
public class CapturePointResponseDto {

	@Builder.Default
	private final String type = "FeatureCollection";
	private final List<FeatureDto> features;
	
}
