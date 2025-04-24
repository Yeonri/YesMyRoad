package com.b201.api.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {

	@Bean
	public RestClient addressRestClient() {
		return RestClient.builder()
			.baseUrl("https://api.vworld.kr")
			.build();
	}
}
