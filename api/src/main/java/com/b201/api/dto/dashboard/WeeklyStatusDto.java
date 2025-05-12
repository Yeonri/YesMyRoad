package com.b201.api.dto.dashboard;

import java.time.LocalDate;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 한 주치(월요일 시작) 파손 건수 합계 + 전주 대비 증감율
 */
@Getter
@AllArgsConstructor
public class WeeklyStatusDto {
	private LocalDate weekStart;  // 주 시작일(월요일)
	private long count;           // 해당 주 파손 건수 합계
	private double changeRate;    // 전주 대비 증감율 (%)
}

