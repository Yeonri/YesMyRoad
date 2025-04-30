package com.b201.api.service;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.b201.api.domain.User;
import com.b201.api.dto.SignupDto;
import com.b201.api.exception.DuplicateUsernameException;
import com.b201.api.repository.UserRepository;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
@Transactional
public class UserService {

	private final UserRepository userRepository;
	private final PasswordEncoder passwordEncoder;

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
}
