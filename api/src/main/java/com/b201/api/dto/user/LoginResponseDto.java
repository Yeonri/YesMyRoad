package com.b201.api.dto.user;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Builder
public class LoginResponseDto {
	private String accessToken;
	private String refreshToken;
	private String userId;
}
