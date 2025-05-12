package com.b201.api.dto.dashboard;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 구 단위 파손 건수
 */
@Getter
@AllArgsConstructor
public class RegionCountDto {
	private String regionName;  // ex) "유성구"
	private long count;         // 파손 건수
}
