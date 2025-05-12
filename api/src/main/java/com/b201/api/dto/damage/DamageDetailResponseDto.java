package com.b201.api.dto.damage;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class DamageDetailResponseDto {

	private String imageUrl;
	private Double risk;
	private List<DamageDto> damages;

}
