package com.b201.api.service;

import java.util.concurrent.TimeUnit;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.b201.api.domain.User;
import com.b201.api.dto.LoginRequestDto;
import com.b201.api.dto.LoginResponseDto;
import com.b201.api.dto.SignupDto;
import com.b201.api.exception.DuplicateUsernameException;
import com.b201.api.repository.UserRepository;
import com.b201.api.util.JwtUtil;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
@Transactional
public class UserService {

	private final UserRepository userRepository;
	private final PasswordEncoder passwordEncoder;
	private final JwtUtil jwtUtil;
	private final StringRedisTemplate stringRedisTemplate;

	//회원가입 로직
	public void signUp(SignupDto signupDto) {
		if (userRepository.findByUsername(signupDto.getId()).isPresent()) {
			throw new DuplicateUsernameException("중복된 ID입니다: " + signupDto.getId());
		}

		User user = User.builder()
			.username(signupDto.getId())
			.password(passwordEncoder.encode(signupDto.getPassword()))
			.name(signupDto.getName())
			.build();

		userRepository.save(user);
	}


	//로그인 로직
	public LoginResponseDto login(LoginRequestDto loginDto) {
		User user = userRepository.findByUsername(loginDto.getId())
			.orElseThrow(() -> new UsernameNotFoundException("존재하지 않는 사용자입니다."));

		if (!passwordEncoder.matches(loginDto.getPassword(), user.getPassword())) {
			throw new BadCredentialsException("비밀번호가 일치하지 않습니다.");
		}

		String accessToken = jwtUtil.generateAccessToken(loginDto.getId());
		String refreshToekn = jwtUtil.generateRefreshToken(loginDto.getId());

		//redis에 refreshToken 저장 (유효기간 2시간)
		stringRedisTemplate.opsForValue().set("RT:"+user.getId(), refreshToekn,2, TimeUnit.HOURS);
		return LoginResponseDto.builder()
			.accessToken(accessToken)
			.refreshToken(refreshToekn)
			.username(user.getUsername())
			.build();
	}


	//AccessToken 재발급 로직
	public LoginResponseDto refreshAccessToken(String requestHeader) {
		if(requestHeader == null || !requestHeader.startsWith("Bearer ")) {
			throw new BadCredentialsException("Refresh Token이 존재하지 않거나 형식이 잘못되었습니다.");
		}

		String refreshToken = requestHeader.substring(7);
		String userId = jwtUtil.getUsernameFromToken(refreshToken);

		String savedRefreshToken = stringRedisTemplate.opsForValue().get("RT:"+userId);
		if(savedRefreshToken == null || !savedRefreshToken.equals(refreshToken)) {
			throw new BadCredentialsException("Refresh Token이 유효하지 않거나 만료되었습니다.");
		}

		String newAccessToken = jwtUtil.generateAccessToken(userId);

		return LoginResponseDto.builder()
			.accessToken(newAccessToken)
			.refreshToken(refreshToken)
			.username(userId)
			.build();
	}

	//로그아웃 로직
	public void logout(String requestHeader) {
		if(requestHeader == null || !requestHeader.startsWith("Bearer ")) {
			throw new BadCredentialsException("AccessToken이 존재하지 않거나 형식이 잘못되었습니다.");
		}

		String accessToken = requestHeader.substring(7);
		String userId = jwtUtil.getUsernameFromToken(accessToken);

		//RefreshToken 삭제
		stringRedisTemplate.delete("RT:"+userId);

		//AccessToken 블랙리스트 등록 (남은 유효 시간만큼 유지)
		long remainTime = jwtUtil.getExpirationTime(accessToken);
		stringRedisTemplate.opsForValue().set("BL:"+accessToken, "logout", remainTime, TimeUnit.MILLISECONDS);
	}
}
