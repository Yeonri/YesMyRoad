package com.b201.api.dto.maintenance;

import java.time.YearMonth;

import lombok.Getter;

@Getter
public class MonthlyMaintenanceStatusDto {

	private final YearMonth month;

	private final long reported;

	private final long received;

	private final long inProgress;

	private final long completed;

	// YearMonth 객체로 받기 위한 생성자
	public MonthlyMaintenanceStatusDto(int year, int month, long reported, long received, long inProgress,
		long completed) {
		this.month = YearMonth.of(year, month);
		this.reported = reported;
		this.received = received;
		this.inProgress = inProgress;
		this.completed = completed;
	}
}
