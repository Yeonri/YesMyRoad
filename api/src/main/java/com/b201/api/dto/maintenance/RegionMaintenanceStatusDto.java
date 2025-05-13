package com.b201.api.dto.maintenance;

import com.fasterxml.jackson.annotation.JsonUnwrapped;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class RegionMaintenanceStatusDto {

	private String name;

	@JsonUnwrapped
	private MaintenanceStatusDto maintenanceStatus;
}
