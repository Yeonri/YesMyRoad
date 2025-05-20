package com.b201.api.dto.dashboard;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

@Getter
@AllArgsConstructor
@ToString
public class RiskStatusDto {
	private long critical; //심각
	private long highRisk; //위험
	private long warning; //주의
	private long safe; //안전
}
