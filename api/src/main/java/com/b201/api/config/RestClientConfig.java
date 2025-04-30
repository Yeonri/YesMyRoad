package com.b201.api.config;

import org.locationtech.jts.geom.GeometryFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class RestClientConfig {

	@Bean
	public GeometryFactory geometryFactory() {
		return new GeometryFactory();
	}

	@Bean
	public RestClient addressRestClient() {
		return RestClient.builder()
			.baseUrl("https://api.vworld.kr")
			.build();
	}
}
