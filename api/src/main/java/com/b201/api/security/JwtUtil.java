package com.b201.api.security;

import java.security.Key;
import java.util.Date;

import org.springframework.stereotype.Component;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Component
@RequiredArgsConstructor
@Slf4j
public class JwtUtil {

	private final JwtProperties jwtProperties;
	private static final long EXPIRATION_TIME = 30 * 60 * 1000; // 30분 (ms 단위)
	private static final long EXPIRATION_REFRESH_TIME = 14 * 24 * 60 * 60 * 1000; // 2주

	// secretKey 생성 메서드 token 생성,검증에 모두 사용됨.
	private Key getSigningKey() {
		log.trace("[getSigningKey] JWT 서명 키 생성");
		return Keys.hmacShaKeyFor(jwtProperties.getSecret().getBytes());
	}

	// jwt token 생성 메서드
	public String generateToken(String username, long expiration) {
		log.info("[generateToken] 토큰 생성 시작, username={}, expiration(ms)={}", username, expiration);
		String token = Jwts.builder()
			.setSubject(username)
			.setExpiration(new Date(System.currentTimeMillis() + expiration))
			.signWith(getSigningKey(), SignatureAlgorithm.HS256)
			.compact();
		log.debug("[generateToken] 생성된 토큰: {}", token);
		return token;
	}

	// jwt access token 생성 메서드
	public String generateAccessToken(String username) {
		log.info("[generateAccessToken] Access Token 생성, username={}", username);
		String accessToken = generateToken(username, EXPIRATION_TIME);
		log.debug("[generateAccessToken] Access Token 생성 완료");
		return accessToken;
	}

	// jwt refresh token 생성 메서드
	public String generateRefreshToken(String username) {
		log.info("[generateRefreshToken] Refresh Token 생성, username={}", username);
		String refreshToken = generateToken(username, EXPIRATION_REFRESH_TIME);
		log.debug("[generateRefreshToken] Refresh Token 생성 완료");
		return refreshToken;
	}

	// jwt 토큰 검증 메서드
	public String getUsernameFromToken(String token) {
		log.debug("[getUsernameFromToken] 토큰 해석 시작: {}", token);
		String username = Jwts.parserBuilder()
			.setSigningKey(getSigningKey())
			.build()
			.parseClaimsJws(token)
			.getBody()
			.getSubject();
		log.info("[getUsernameFromToken] 토큰에서 추출된 username={}", username);
		return username;
	}

	public long getExpirationTime(String token) {
		log.debug("[getExpirationTime] 만료시간 계산 시작, token={}", token);
		Date expiration = Jwts.parserBuilder()
			.setSigningKey(getSigningKey())
			.build()
			.parseClaimsJws(token)
			.getBody()
			.getExpiration();

		long remain = expiration.getTime() - System.currentTimeMillis();
		log.info("[getExpirationTime] 남은 만료시간(ms)={}", remain);
		return remain;
	}
}
