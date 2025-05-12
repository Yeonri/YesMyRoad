package com.b201.api.dto.maintenance;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

// 일간, 주간, 월간 보수공사 완료 개수
@Getter
@AllArgsConstructor
@ToString
public class CompletionStatsDto {

	long daily;

	long weekly;

	long monthly;
}
