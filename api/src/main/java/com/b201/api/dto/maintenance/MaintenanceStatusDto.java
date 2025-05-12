package com.b201.api.dto.maintenance;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

// 전체 보수공사(수리상태) 미완료/진행중/완료 개수
@ToString
@Getter
@AllArgsConstructor
public class MaintenanceStatusDto {

	long reported;

	long received;

	long inProgress;

	long completed;
}
