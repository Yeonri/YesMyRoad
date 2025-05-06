package com.b201.api.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Getter;

@Getter
public class LoginRequestDto {

	@NotNull(message = "id는 필수값입니다.")
	private String id;

	@NotNull(message = "password는 필수값입니다.")
	private String password;
}
