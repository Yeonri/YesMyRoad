package com.b201.api.exception;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

	@Override
	protected ResponseEntity<Object> handleMethodArgumentNotValid(MethodArgumentNotValidException ex,
		HttpHeaders headers, HttpStatusCode status, WebRequest request) {
		List<ValidationError> errors = ex.getBindingResult()
			.getFieldErrors()
			.stream()
			.map(fe -> new ValidationError(fe.getField(), fe.getDefaultMessage()))
			.collect(Collectors.toList());

		return ResponseEntity.badRequest().body(new ErrorResponse(errors));
	}

	@ExceptionHandler(AddressLookupException.class)
	public ResponseEntity<ErrorResponse> handleAddressLookup(AddressLookupException ex) {
		return ResponseEntity
			.status(HttpStatus.BAD_REQUEST)
			.body(new ErrorResponse(ex.getMessage()));
	}

	public static record ValidationError(String field, String defaultMessage) {
	}

	public static record ErrorResponse(List<ValidationError> errors) {
		public ErrorResponse(String message) {
			this(List.of(new ValidationError(null, message)));
		}
	}
}


