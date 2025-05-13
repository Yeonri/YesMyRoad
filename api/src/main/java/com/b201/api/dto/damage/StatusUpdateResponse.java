package com.b201.api.dto.damage;

import com.b201.api.domain.CaptureDamage;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
@AllArgsConstructor
public class StatusUpdateResponse {

	private Integer damageId;
	private CaptureDamage.DamageStatus damageStatus;
}
