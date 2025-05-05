package com.b201.api.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.LoginRequestDto;
import com.b201.api.dto.SignupDto;
import com.b201.api.service.UserService;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/users")
public class UserController {

	private final UserService userService;

	@PostMapping("/signup")
	public ResponseEntity<?> signup(@Valid @RequestBody SignupDto signupDto) {
		userService.signUp(signupDto);
		return ResponseEntity.ok().build();
	}

	@PostMapping("/login")
	public ResponseEntity<?> login(@Valid @RequestBody LoginRequestDto loginDto) {
		return ResponseEntity.ok().body(userService.login(loginDto));
	}

	@PostMapping("/refresh")
	public ResponseEntity<?> refresh(HttpServletRequest request) {
		String requestHeader = request.getHeader("Authorization");

		return ResponseEntity.ok().body(userService.refreshAccessToken(requestHeader));
	}
}
