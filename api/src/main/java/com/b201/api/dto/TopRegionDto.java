package com.b201.api.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class TopRegionDto {
	private final String regionName;
	private final long totalCount;
	private final long crackCount;
	private final long holeCount;
}
