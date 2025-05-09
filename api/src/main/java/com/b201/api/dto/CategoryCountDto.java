package com.b201.api.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

//유형별 도로파손 분포
@Getter
@AllArgsConstructor
public class CategoryCountDto {
	private String categoryName;
	private long count;
}
