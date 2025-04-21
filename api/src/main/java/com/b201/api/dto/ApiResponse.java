package com.b201.api.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
public class ApiResponse<T> {

	private final int statusCode;
	private final String message;
	private final T data;

	@Builder
	public ApiResponse(int statusCode, String message, T data) {
		this.statusCode = statusCode;
		this.message = message;
		this.data = data;
	}

}
