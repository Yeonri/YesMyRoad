package com.b201.api.dto.dashboard;

import java.time.YearMonth;

import lombok.Getter;

/**
 * 월별 파손 유형별 건수 + 총계
 */
@Getter
public class MonthlyDamageSummaryDto {
	private final YearMonth month;     // 2025-05 형태
	private final long crackCount;     // 도로균열 건수
	private final long holeCount;      // 도로홀 건수
	private final long totalCount;     // 전체 파손 건수

	// JPQL new 연산자를 위해 year, month를 int로 받아서 YearMonth로 변환
	public MonthlyDamageSummaryDto(int year,
		int month,
		long crackCount,
		long holeCount,
		long totalCount) {
		this.month = YearMonth.of(year, month);
		this.crackCount = crackCount;
		this.holeCount = holeCount;
		this.totalCount = totalCount;
	}
}
