package com.b201.api.service;

import java.util.concurrent.TimeUnit;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.b201.api.domain.Region;
import com.b201.api.domain.User;
import com.b201.api.dto.user.DuplicateIdDto;
import com.b201.api.dto.user.LoginRequestDto;
import com.b201.api.dto.user.LoginResponseDto;
import com.b201.api.dto.user.SignupDto;
import com.b201.api.exception.DuplicateUsernameException;
import com.b201.api.repository.RegionRepository;
import com.b201.api.repository.UserRepository;
import com.b201.api.security.JwtUtil;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional
public class UserService {

	private final UserRepository userRepository;
	private final PasswordEncoder passwordEncoder;
	private final JwtUtil jwtUtil;
	private final StringRedisTemplate stringRedisTemplate;
	private final RegionRepository regionRepository;

	// 회원가입 로직
	public void signUp(SignupDto signupDto) {
		log.info("[signUp] 호출됨, username={}, regionId={}", signupDto.getId(), signupDto.getRegionId());

		if (userRepository.findByUsername(signupDto.getId()).isPresent()) {
			log.error("[signUp] 중복된 ID: {}", signupDto.getId());
			throw new DuplicateUsernameException("중복된 ID입니다: " + signupDto.getId());
		}

		Region region = regionRepository.findById(signupDto.getRegionId())
			.orElseThrow(() -> {
				log.error("[signUp] 존재하지 않는 regionId: {}", signupDto.getRegionId());
				return new IllegalArgumentException("해당 지역이 존재하지 않습니다.");
			});

		User user = User.builder()
			.username(signupDto.getId())
			.password(passwordEncoder.encode(signupDto.getPassword()))
			.name(signupDto.getName())
			.region(region)
			.build();

		userRepository.save(user);
		log.info("[signUp] 회원가입 완료, username={}", signupDto.getId());
	}

	// 아이디 중복 체크
	public DuplicateIdDto isDuplicateUsername(String username) {
		log.info("[isDuplicateUsername] 호출됨, username={}", username);
		boolean available = userRepository.findByUsername(username).isEmpty();
		log.debug("[isDuplicateUsername] available={}", available);
		return DuplicateIdDto.builder()
			.available(available ? 1 : 0)
			.build();
	}

	// 로그인 로직
	public LoginResponseDto login(LoginRequestDto loginDto) {
		log.info("[login] 호출됨, username={}", loginDto.getId());

		User user = userRepository.findByUsername(loginDto.getId())
			.orElseThrow(() -> {
				log.error("[login] 존재하지 않는 사용자: {}", loginDto.getId());
				return new UsernameNotFoundException("존재하지 않는 사용자입니다.");
			});

		if (!passwordEncoder.matches(loginDto.getPassword(), user.getPassword())) {
			log.error("[login] 비밀번호 불일치, username={}", loginDto.getId());
			throw new BadCredentialsException("비밀번호가 일치하지 않습니다.");
		}

		String accessToken = jwtUtil.generateAccessToken(loginDto.getId());
		String refreshToken = jwtUtil.generateRefreshToken(loginDto.getId());
		log.debug("[login] accessToken 발급, username={}", loginDto.getId());

		// redis에 refreshToken 저장 (유효기간 2시간)
		stringRedisTemplate.opsForValue()
			.set("RT:" + user.getUsername(), refreshToken, 2, TimeUnit.HOURS);
		log.info("[login] 로그인 성공, username={}", loginDto.getId());

		return LoginResponseDto.builder()
			.accessToken(accessToken)
			.refreshToken(refreshToken)
			.userId(user.getUsername())
			.build();
	}

	// AccessToken 재발급 로직
	public LoginResponseDto refreshAccessToken(String requestHeader) {
		log.info("[refreshAccessToken] 호출됨");

		if (requestHeader == null || !requestHeader.startsWith("Bearer ")) {
			log.error("[refreshAccessToken] 잘못된 헤더: {}", requestHeader);
			throw new BadCredentialsException("Refresh Token이 존재하지 않거나 형식이 잘못되었습니다.");
		}

		String refreshToken = requestHeader.substring(7);
		String userId = jwtUtil.getUsernameFromToken(refreshToken);
		log.debug("[refreshAccessToken] token 파싱, userId={}", userId);

		String saved = stringRedisTemplate.opsForValue().get("RT:" + userId);
		if (saved == null || !saved.equals(refreshToken)) {
			log.error("[refreshAccessToken] 유효하지 않은 Refresh Token, userId={}", userId);
			throw new BadCredentialsException("Refresh Token이 유효하지 않거나 만료되었습니다.");
		}

		String newAccess = jwtUtil.generateAccessToken(userId);
		log.info("[refreshAccessToken] AccessToken 재발급, userId={}", userId);

		return LoginResponseDto.builder()
			.accessToken(newAccess)
			.refreshToken(refreshToken)
			.userId(userId)
			.build();
	}

	// 로그아웃 로직
	public void logout(String requestHeader) {
		log.info("[logout] 호출됨");

		if (requestHeader == null || !requestHeader.startsWith("Bearer ")) {
			log.error("[logout] 잘못된 헤더: {}", requestHeader);
			throw new BadCredentialsException("AccessToken이 존재하지 않거나 형식이 잘못되었습니다.");
		}

		String accessToken = requestHeader.substring(7);
		String userId;

		try {
			userId = jwtUtil.getUsernameFromToken(accessToken);
		} catch (Exception e) {
			log.error("[logout] 유효하지 않은 AccessToken: {}", accessToken, e);
			throw new BadCredentialsException("AccessToken이 유효하지 않거나 만료되었습니다.");
		}

		// RefreshToken 삭제
		stringRedisTemplate.delete("RT:" + userId);
		log.debug("[logout] RefreshToken 삭제, userId={}", userId);

		// AccessToken 블랙리스트 등록 (남은 유효 시간만큼 유지)
		long remain = jwtUtil.getExpirationTime(accessToken);
		stringRedisTemplate.opsForValue()
			.set("BL:" + accessToken, "logout", remain, TimeUnit.MILLISECONDS);
		log.info("[logout] 로그아웃 완료, userId={}", userId);
	}
}