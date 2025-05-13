package com.b201.api.dto.damage;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class StatusUpdateRequest {
	/**
	 * DamageStatus enum 이름(REPORTED, RECEIVED, IN_PROGRESS, COMPLETED)
	 */
	private String status;
}
