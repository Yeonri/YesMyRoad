package com.b201.api.dto.dashboard;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class RegionNameWithCountDto {

	private Integer regionId;

	private List<RegionCountDto> destrictions;
}
