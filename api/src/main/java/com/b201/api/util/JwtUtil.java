package com.b201.api.util;

import org.springframework.stereotype.Component;

import com.b201.api.config.JwtProperties;

import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class JwtUtil {

	private final JwtProperties jwtProperties;
	private static final long EXPIRATION_TIME = 60 * 60 * 1000; // 1시간 (ms 단위)

}
