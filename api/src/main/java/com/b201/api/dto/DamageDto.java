package com.b201.api.dto;

import java.time.LocalDateTime;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class DamageDto {

	private Integer id;
	private String category;
	private int status;
	private LocalDateTime updatedAt;
}
