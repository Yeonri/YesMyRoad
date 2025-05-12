package com.b201.api.dto.maintenance;

import lombok.AllArgsConstructor;
import lombok.Getter;

// 전체 보수공사(수리상태) 미완료/진행중/완료 개수
@Getter
@AllArgsConstructor
public class MaintenanceStatusDto {

	private long reported;

	private long received;

	private long inProgress;

	private long completed;
}
