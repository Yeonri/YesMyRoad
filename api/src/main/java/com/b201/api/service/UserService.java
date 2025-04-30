package com.b201.api.service;

import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.b201.api.domain.User;
import com.b201.api.dto.LoginDto;
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

	public String login(LoginDto loginDto) {
		User user = userRepository.findByUsername(loginDto.getId())
			.orElseThrow(() -> new UsernameNotFoundException("존재하지 않는 사용자입니다."));

		if (!passwordEncoder.matches(loginDto.getPassword(), user.getPassword())) {
			throw new BadCredentialsException("비밀번호가 일치하지 않습니다.");
		}

		return jwtUtil.generateToken(user.getUsername());
	}
}
