package com.b201.api.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Getter;

@Getter
public class SignupDto {

	@NotNull(message = "id는 필수값입니다.")
	private String id;

	@NotNull(message = "password는 필수값입니다.")
	private String password;

	@NotNull(message = "name은 필수값입니다.")
	private String name;

	@NotNull(message = "regionId는 필수값입니다.")
	private Integer regionId;
}
