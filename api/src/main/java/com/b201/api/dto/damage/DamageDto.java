package com.b201.api.dto.damage;

import java.time.LocalDateTime;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

@Getter
@Builder
@ToString
@AllArgsConstructor
public class DamageDto {

	private Integer id;
	private String category;
	private int status;
	private LocalDateTime updatedAt;
}
