package com.b201.api.security;

import java.security.Key;
import java.util.Date;

import org.springframework.stereotype.Component;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;

@Component
@RequiredArgsConstructor
@Log4j2
public class JwtUtil {

	private final JwtProperties jwtProperties;
	private static final long EXPIRATION_TIME = 30 * 60 * 1000; // 30분 (ms 단위)
	private static final long EXPIRATION_REFRESH_TIME = 2 * 60 * 60 * 1000; // 2시간

	//secretKey 생성 메서드 token 생성,검증에 모두 사용됨.
	private Key getSigningKey() {
		//jwtProperties에서 환경변수로 주입받은 secretkey를 가져다가 생성함.
		return Keys.hmacShaKeyFor(jwtProperties.getSecret().getBytes());
	}

	//jwt token 생성 메서드
	public String generateToken(String username, long expiration) {
		return Jwts.builder()
			//이 토큰의 주인이 누구인지 사용자 정보 저장
			.setSubject(username)
			//만료 시간 설정
			.setExpiration(new Date(System.currentTimeMillis() + expiration))
			//인증 방법 설정
			.signWith(getSigningKey(),
				SignatureAlgorithm.HS256)
			//생성
			.compact();
	}

	//jwt access token 생성 메서드
	public String generateAccessToken(String username) {
		return generateToken(username, EXPIRATION_TIME);
	}

	//jwt refresh token 생성 메서드
	public String generateRefreshToken(String username) {
		return generateToken(username, EXPIRATION_REFRESH_TIME);
	}

	//jwt 토큰 검증 메서드
	public String getUsernameFromToken(String token) {
		return Jwts.parserBuilder()
			.setSigningKey(getSigningKey()) //서명 검증용 키 설정
			.build() //jwt 분석 도구 생성
			.parseClaimsJws(token) //실제 토큰 검증
			.getBody()
			.getSubject(); //subject (id) 추출
	}

	public long getExpirationTime(String token) {
		Date expiration = Jwts.parserBuilder()
			.setSigningKey(getSigningKey())
			.build()
			.parseClaimsJws(token)
			.getBody()
			.getExpiration();

		return expiration.getTime() - System.currentTimeMillis();
	}
}
