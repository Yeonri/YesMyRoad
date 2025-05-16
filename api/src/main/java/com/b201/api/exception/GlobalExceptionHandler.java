package com.b201.api.exception;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import lombok.extern.slf4j.Slf4j;

@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

	// Validation 오류 처리 (예: @Valid 실패)
	@Override
	protected ResponseEntity<Object> handleMethodArgumentNotValid(MethodArgumentNotValidException ex,
		HttpHeaders headers, HttpStatusCode status, WebRequest request) {

		log.warn("[handleMethodArgumentNotValid] validation failed: {}", ex.getBindingResult());
		List<ValidationError> errors = ex.getBindingResult()
			.getFieldErrors()
			.stream()
			.map(fe -> new ValidationError(fe.getField(), fe.getDefaultMessage()))
			.collect(Collectors.toList());

		return ResponseEntity.badRequest().body(new ErrorResponse(errors));
	}

	// 사용자 존재하지 않음
	@ExceptionHandler(UsernameNotFoundException.class)
	public ResponseEntity<ErrorResponse> handleUsernameNotFound(UsernameNotFoundException ex) {
		log.warn("[handleUsernameNotFound] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.UNAUTHORIZED)
			.body(new ErrorResponse(ex.getMessage()));
	}

	// 비밀번호 불일치 등 인증 실패
	@ExceptionHandler(BadCredentialsException.class)
	public ResponseEntity<ErrorResponse> handleBadCredentials(BadCredentialsException ex) {
		log.warn("[handleBadCredentials] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.UNAUTHORIZED)
			.body(new ErrorResponse(ex.getMessage()));
	}

	// 중복 ID
	@ExceptionHandler(DuplicateUsernameException.class)
	public ResponseEntity<ErrorResponse> handleDuplicateUsername(DuplicateUsernameException ex) {
		log.warn("[handleDuplicateUsername] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.BAD_REQUEST)
			.body(new ErrorResponse(ex.getMessage()));
	}

	// 주소 조회 실패
	@ExceptionHandler(AddressLookupException.class)
	public ResponseEntity<ErrorResponse> handleAddressLookup(AddressLookupException ex) {
		log.warn("[handleAddressLookup] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.BAD_REQUEST)
			.body(new ErrorResponse(ex.getMessage()));
	}

	//회원가입 시 잘못된 regionId를 매칭했을 때
	@ExceptionHandler(IllegalArgumentException.class)
	public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
		log.warn("[handleIllegalArgument] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.BAD_REQUEST)
			.body(new ErrorResponse(ex.getMessage()));
	}

	@ExceptionHandler(ArithmeticException.class)
	public ResponseEntity<Void> handleArithmetic(ArithmeticException ex) {
		log.warn("[handleArithmetic] {}", ex.getMessage());
		return ResponseEntity
			.status(HttpStatus.NO_CONTENT)
			.build();
	}

	// 기본 포맷
	public static record ValidationError(String field, String defaultMessage) {
	}

	public static record ErrorResponse(List<ValidationError> errors) {
		public ErrorResponse(String message) {
			this(List.of(new ValidationError(null, message)));
		}
	}
}



