package com.b201.api.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class DuplicateIdDto {
	boolean available;
}
