package com.b201.api.dto;

import java.time.LocalDate;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 하루치 집계 + 전일 대비 증감율
 */
@Getter
@AllArgsConstructor
public class DailyStatusDto {
	private LocalDate day;     // 날짜
	private long count;        // 해당 날짜 총 파손 건수
	private double changeRate; // 전일 대비 증감율 (%) — 전일이 0이면 0.0 처리
}
