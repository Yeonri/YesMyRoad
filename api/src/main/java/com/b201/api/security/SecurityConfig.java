package com.b201.api.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import lombok.RequiredArgsConstructor;

@Configuration
@RequiredArgsConstructor
public class SecurityConfig {

	private final JwtAuthenticationFilter jwtAuthenticationFilter;

	//비밀번호 암호용
	@Bean
	public PasswordEncoder passwordEncoder() {
		return new BCryptPasswordEncoder();
	}

	@Bean
	public CorsConfigurationSource corsConfigurationSource() {
		CorsConfiguration config = new CorsConfiguration();
		config.addAllowedOriginPattern("https://k12b201.p.ssafy.io");
		config.addAllowedOriginPattern("http://localhost:3000");
		config.addAllowedMethod("GET");
		config.addAllowedMethod("POST");
		config.addAllowedMethod("PUT");
		config.addAllowedMethod("DELETE");
		config.addAllowedHeader("Content-Type");
		config.addAllowedHeader("Authorization");
		config.setAllowCredentials(true);

		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
		source.registerCorsConfiguration("/**", config);
		return source;
	}

	@Bean
	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		http
			.cors(cors -> cors.configurationSource(corsConfigurationSource())) //cors 설정
			.csrf(AbstractHttpConfigurer::disable) //jwt 인증방식은 세션 인증 방식이 아니므로 csrf 설정이 불필요
			.authorizeHttpRequests(auth -> auth
				.requestMatchers("/api/users/**", "/api/detect").permitAll()
				.anyRequest().authenticated())

			// JwtAuthenticationFilter를 UsernamePasswordAuthenticationFilter(Spring Security 기본 로그인 필터) 앞에 등록
			.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

		return http.build();
	}
}
