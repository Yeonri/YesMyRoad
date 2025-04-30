package com.b201.api.filter;

import java.io.IOException;
import java.util.Collections;

import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.web.filter.OncePerRequestFilter;

import com.b201.api.util.JwtUtil;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

	private final JwtUtil jwtUtil;

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
		FilterChain filterChain) throws ServletException, IOException {

		//Authorization 헤더 추출
		String authHeader = request.getHeader("Authorization");

		//헤더가 있고 Bearer로 시작하는지 확인
		if (authHeader != null && authHeader.startsWith("Bearer ")) {
			String token = authHeader.substring(7); //Bearer 이후 실제 토큰 값

			try{
				//추출한 토큰 값을 검증해서 해당 user의 id를 추출
				String username = jwtUtil.getUsernameFromToken(token);

				//username이 존재하고 아직 인증되지 않은 경우
				if(username!=null && SecurityContextHolder.getContext().getAuthentication()==null){

					//사용자 인증 객체 생성 (첫번쨰 인자 : 로그인한 사용자 이름, 두번째 인자 : 인증수단(여기선 password인데 jwt로 인증했기 때문에 null, 세번쨰 인자 : 권한)
					UsernamePasswordAuthenticationToken authentication =
						new UsernamePasswordAuthenticationToken(username, null, Collections.emptyList());

					//현재 요청 정보를 사용자 인증 객체에 포함시킴 (ip줏고, 세션 id 등) (보안 로깅에 사용됨)
					authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));

					//SecurityContext에 인증 객체 등록
					SecurityContextHolder.getContext().setAuthentication(authentication);

				}
			}catch(Exception e){
				//jwt가 유효하지 않거나 존재하지 않는경우 401 반환
				response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
				response.getWriter().write("jwt 토큰 값이 유효하지 않습니다.");
				return;
			}

		}

		filterChain.doFilter(request, response);
	}

}
