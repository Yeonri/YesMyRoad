package com.b201.api.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.b201.api.dto.user.LoginRequestDto;
import com.b201.api.dto.user.SignupDto;
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
		return ResponseEntity.status(HttpStatus.CREATED).build();
	}

	@GetMapping(
		path = "/id",
		produces = MediaType.APPLICATION_JSON_VALUE
	)
	public ResponseEntity<?> getUser(@RequestParam String userId) {
		return ResponseEntity.ok().body(userService.isDuplicateUsername(userId));
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

	@PostMapping("/logout")
	public ResponseEntity<?> logout(HttpServletRequest request) {
		String requestHeader = request.getHeader("Authorization");
		userService.logout(requestHeader);
		return ResponseEntity.ok().body("로그아웃 성공");
	}
}
