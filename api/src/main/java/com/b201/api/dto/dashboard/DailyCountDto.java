package com.b201.api.dto.dashboard;

import java.sql.Date;
import java.time.LocalDate;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class DailyCountDto {
	private LocalDate day;
	private long count;

	/**
	 * JPQL DATE() 함수는 java.sql.Date 를 반환하므로,
	 * 이 생성자를 추가해야 'new DailyCountDto(java.sql.Date, Long)' 호출이 성공합니다.
	 */
	public DailyCountDto(Date day, Long count) {
		this.day = day == null ? null : day.toLocalDate();
		this.count = (count != null ? count : 0L);
	}
}
