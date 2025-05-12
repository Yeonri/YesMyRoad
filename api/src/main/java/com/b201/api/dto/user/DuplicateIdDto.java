package com.b201.api.dto.user;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class DuplicateIdDto {
	int available;
}
