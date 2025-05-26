package com.b201.api.dto.dashboard;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 구 단위 파손 건수
 */
@Getter
@AllArgsConstructor
public class RegionCountDto {
	private String name;  // ex) "유성구"
	private long value;         // 파손 건수
}
