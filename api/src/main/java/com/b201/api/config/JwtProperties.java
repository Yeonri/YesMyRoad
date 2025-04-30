package com.b201.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import lombok.Getter;
import lombok.Setter;

/**
 * jwt 관련 환경 변수들을 관리하는 class 입니다.
 */
@ConfigurationProperties(prefix = "jwt")
@Component
@Getter
@Setter // configurationProperties 어노테이션은 기본적으로 yml 파일에 있는 값을 setter를 통해 주입하려고 한다.
public class JwtProperties {

	private String secret;
}
