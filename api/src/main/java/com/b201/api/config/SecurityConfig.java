package com.b201.api.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import com.b201.api.filter.JwtAuthenticationFilter;
import com.b201.api.util.JwtUtil;

import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class SecurityConfig {

	private final JwtUtil jwtUtil;

	//비밀번호 암호용
	@Bean
	public PasswordEncoder passwordEncoder() {
		return new BCryptPasswordEncoder();
	}

	@Bean
	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		http
			.cors(cors -> {}) //cors 설정 가능
			.csrf(AbstractHttpConfigurer::disable) //jwt 인증방싟은 세션 인증 방식이 아니므로 csrf 설정이 불필요
			.authorizeHttpRequests(auth -> auth
				.requestMatchers("/api/users/**").permitAll()
				.anyRequest().authenticated())

			// JwtAuthenticationFilter를 UsernamePasswordAuthenticationFilter(Spring Security 기본 로그인 필터) 앞에 등록
			.addFilterBefore(new JwtAuthenticationFilter(jwtUtil), UsernamePasswordAuthenticationFilter.class);

		return http.build();
	}
}
