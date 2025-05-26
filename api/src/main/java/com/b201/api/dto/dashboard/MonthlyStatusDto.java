package com.b201.api.dto.dashboard;

import java.time.YearMonth;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 한 달치(YearMonth) 파손 건수 합계 + 전월 대비 증감율
 */
@Getter
@AllArgsConstructor
public class MonthlyStatusDto {
	private YearMonth month;      // 해당 월 (예: 2025-05)
	private long count;           // 해당 월 파손 건수 합계
	private double changeRate;    // 전월 대비 증감율 (%)
}
