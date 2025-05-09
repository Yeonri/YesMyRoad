package com.b201.api.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class StatusUpdateRequest {
	/**
	 * DamageStatus enum 이름(REPORTED, RECEIVED, IN_PROGRESS, COMPLETED)
	 */
	private String status;
}
